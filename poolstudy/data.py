from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from .common import ROOT, write_json

# Fixed constants; no test-set statistics are fitted.
STATS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
         "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))}

def keyed_rng(*parts):
    b = "|".join(str(p) for p in parts).encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little"))

def load_raw(cfg, train=True, download=False):
    if cfg["synthetic"]:
        nc = 10 if cfg["dataset"] == "cifar10" else 100
        count = cfg["synthetic_train_per_class"] if train else cfg["synthetic_test_per_class"]
        y = np.repeat(np.arange(nc), count)
        # Random images and labels are explicitly for engineering checks only.
        x = keyed_rng("SYNTHETIC", train).integers(0, 256, (len(y), 32, 32, 3), dtype=np.uint8)
        return x, y, [f"synthetic_{i}" for i in range(nc)]
    path = Path(cfg["data_dir"])
    if not path.is_absolute():
        path = ROOT / path
    cls = CIFAR10 if cfg["dataset"] == "cifar10" else CIFAR100
    ds = cls(str(path), train=train, download=download)
    return ds.data, np.asarray(ds.targets, dtype=np.int64), ds.classes

def make_split(y, cfg, out=None):
    rng = np.random.default_rng(cfg["split_seed"])
    train, valid = [], []
    for c in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == c))
        nv = max(1, round(len(ids) * cfg["validation_fraction"]))
        if nv >= len(ids):
            raise ValueError("Too few training samples per class")
        valid.extend(ids[:nv].tolist())
        train.extend(ids[nv:].tolist())
    train, valid = np.asarray(sorted(train)), np.asarray(sorted(valid))
    if out is not None:
        import pandas as pd
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "split_indices.npz"
        if path.exists():
            with np.load(path) as old:
                if not np.array_equal(train, old["train"]) or not np.array_equal(valid, old["validation"]):
                    raise RuntimeError("Saved split differs from current split")
        else:
            np.savez_compressed(path, train=train, validation=valid)
        pd.DataFrame({"original_id": np.r_[train, valid], "label": y[np.r_[train, valid]],
                      "split": ["train"]*len(train) + ["validation"]*len(valid)}).to_csv(out/"split_manifest.csv", index=False)
        write_json(out/"split_summary.json", {"train": len(train), "validation": len(valid),
            "seed": cfg["split_seed"], "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
            "validation_sha256": hashlib.sha256(valid.tobytes()).hexdigest()})
    return train, valid

def conditions(cfg):
    result = [{"name": "clean", "kind": "clean", "area": 0., "fill": "none", "repeat": 0, "sigma": 0.}]
    for area in cfg["areas"]:
        for fill in cfg["fills"]:
            for repeat in range(cfg["repeats"]):
                result.append({"name": f"occ_{fill}_a{area:g}_r{repeat}", "kind": "occlusion", "area": area,
                               "fill": fill, "repeat": repeat, "sigma": 0.})
    for sigma in cfg["noise_sigmas"]:
        result.append({"name": f"noise_s{sigma:g}", "kind": "gaussian", "area": 0., "fill": "none", "repeat": 0, "sigma": sigma})
    return result

def corrupt(image, original_id, cond, cfg, split="test"):
    """CPU pixel-space corruption. All methods and training seeds share masks.

    The square is completely inside the image. Actual area is quantized by
    round(32*sqrt(requested_area))**2/1024, and is retained in output data.
    mean/noise fills share location. Full nesting across areas is not guaranteed.
    """
    x = np.asarray(image, dtype=np.float32).copy()
    mask = np.zeros(x.shape[:2], dtype=np.float32)
    if cond["kind"] == "occlusion":
        side = min(32, max(1, int(round(32 * math.sqrt(cond["area"])))))
        rng = keyed_rng(cfg["corruption_seed"], split, original_id, cond["repeat"], "location")
        uy, ux = rng.random(2)
        top, left = int(uy * (33-side)), int(ux * (33-side))
        mask[top:top+side, left:left+side] = 1
        if cond["fill"] == "mean":
            value = np.asarray(STATS[cfg["dataset"]][0], dtype=np.float32)
        else:
            value = keyed_rng(cfg["corruption_seed"], split, original_id, cond["repeat"], side, "fill").random((side,side,3)).astype(np.float32)
        x[top:top+side, left:left+side] = value
    elif cond["kind"] == "gaussian":
        noise = keyed_rng(cfg["corruption_seed"], split, original_id, "gaussian").normal(0, cond["sigma"], x.shape)
        x = np.clip(x + noise, 0, 1).astype(np.float32)
    return x, mask

class Images(Dataset):
    def __init__(self, x, y, ids, cfg, *, train=False, seed=0, epoch=0, condition=None, split="test"):
        self.x, self.y, self.ids, self.cfg = x, y, np.asarray(ids), cfg
        self.train, self.seed, self.epoch = train, seed, epoch
        self.condition = condition or conditions(cfg)[0]
        self.split = split
        mean, std = STATS[cfg["dataset"]]
        self.mean = torch.tensor(mean)[:, None, None]
        self.std = torch.tensor(std)[:, None, None]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        original = int(self.ids[index])
        im = self.x[original]
        if self.train:
            rng = keyed_rng(self.seed, self.epoch, original, "augmentation")
            yy, xx = rng.integers(0, 9, 2)
            im = np.pad(im, ((4,4), (4,4), (0,0)), mode="reflect")[yy:yy+32, xx:xx+32]
            if rng.random() < .5:
                im = im[:, ::-1]
        im = im.astype(np.float32) / 255.
        if self.train:
            mask = np.zeros((32,32), dtype=np.float32)
            if self.cfg["train_occlusion_probability"] and rng.random() < self.cfg["train_occlusion_probability"]:
                cond = {"kind": "occlusion", "area": self.cfg["train_occlusion_area"], "fill": "mean", "repeat": self.epoch}
                im, mask = corrupt(im, original, cond, self.cfg, "training_augmentation")
        else:
            im, mask = corrupt(im, original, self.condition, self.cfg, self.split)
        tensor = torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1)))
        return (tensor - self.mean) / self.std, int(self.y[original]), original, torch.from_numpy(mask)

def loader(ds, cfg, train=False, epoch=0, seed=0):
    gen = torch.Generator().manual_seed(seed * 100003 + epoch)
    return DataLoader(ds, batch_size=cfg["batch_size"] if train else cfg["eval_batch_size"],
        shuffle=train, num_workers=cfg["workers"], generator=gen, drop_last=False,
        persistent_workers=False, pin_memory=False)
