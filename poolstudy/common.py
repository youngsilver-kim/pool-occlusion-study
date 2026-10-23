from __future__ import annotations
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gap", "max", "gem", "attention", "attention_entropy", "attention_floor", "mixed_attention")
DEFAULT = {
    "name": "main", "dataset": "cifar100", "backbone": "resnet20", "width": 16,
    "synthetic": False, "epochs": 80, "seeds": [13, 42, 2026], "methods": list(METHODS),
    "batch_size": 128, "eval_batch_size": 256, "workers": 0, "cpu_threads": 4,
    "device": "auto", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0005,
    "warmup_epochs": 5, "label_smoothing": 0.0, "validation_fraction": 0.1,
    "split_seed": 317, "corruption_seed": 911, "entropy_lambda": 0.05,
    "floor_lambda": 1.0, "entropy_floor": 0.8, "mix_alpha": 0.5,
    "train_occlusion_probability": 0.0, "train_occlusion_area": 0.25,
    "areas": [0.1, 0.25, 0.4], "fills": ["mean", "noise"], "repeats": 2,
    "noise_sigmas": [0.03, 0.08, 0.15], "ece_bins": 15,
    "profile_warmup": 30, "profile_steps": 100, "profile_trials": 5,
    "qualitative_ids": [0, 1, 2, 3, 4, 5], "data_dir": "data", "output_dir": "outputs",
    "synthetic_train_per_class": 24, "synthetic_test_per_class": 6,
}

def read_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if unknown := set(raw) - set(DEFAULT):
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    cfg = copy.deepcopy(DEFAULT)
    cfg.update(raw)
    if cfg["dataset"] not in ("cifar10", "cifar100"):
        raise ValueError("dataset must be cifar10 or cifar100")
    if cfg["backbone"] not in ("resnet20", "resnet32"):
        raise ValueError("backbone must be resnet20 or resnet32")
    if not cfg["methods"] or set(cfg["methods"]) - set(METHODS):
        raise ValueError(f"methods must be selected from {METHODS}")
    for key in ("seeds", "methods", "areas", "fills", "noise_sigmas"):
        if len(cfg[key]) != len(set(cfg[key])):
            raise ValueError(f"Duplicate {key}")
    if any(x not in ("mean", "noise") for x in cfg["fills"]):
        raise ValueError("Unknown occlusion fill")
    if not cfg["areas"] or not cfg["fills"]:
        raise ValueError("Specify at least one occlusion area and fill")
    if not cfg["seeds"] or cfg["epochs"] < 1 or cfg["repeats"] < 1:
        raise ValueError("Empty seeds or nonpositive epochs/repeats")
    if not 0 < cfg["validation_fraction"] < 0.5:
        raise ValueError("validation_fraction must lie in (0, .5)")
    if not 0 <= cfg["entropy_floor"] <= 1 or not 0 <= cfg["mix_alpha"] <= 1:
        raise ValueError("entropy_floor and mix_alpha must lie in [0, 1]")
    if any(not 0 < a < 1 for a in cfg["areas"]):
        raise ValueError("Occlusion areas must lie in (0, 1)")
    if not 0 <= cfg["train_occlusion_probability"] <= 1:
        raise ValueError("Invalid training occlusion probability")
    if cfg["entropy_lambda"] < 0 or cfg["floor_lambda"] < 0:
        raise ValueError("Regularization weights must be nonnegative")
    if any(s <= 0 for s in cfg["noise_sigmas"]):
        raise ValueError("Noise sigma must be positive")
    return cfg

def fingerprint(cfg):
    # Device and paths are part of the protocol: changing them requires a new name.
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def experiment_dir(cfg):
    base = Path(cfg["output_dir"])
    if not base.is_absolute():
        base = ROOT / base
    return base / cfg["name"]

def lock_protocol(cfg):
    out = experiment_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "protocol.json"
    if lock.exists() and read_json(lock) != cfg:
        raise RuntimeError(f"Protocol changed for {out}. Use a new config name; never mix runs.")
    if not lock.exists():
        write_json(lock, cfg)
    return out

def choose_device(request="auto"):
    if request == "auto":
        request = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if request == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable. Check Apple Silicon/native Python and PyTorch, or use --device cpu for a check.")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return torch.device(request)

def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    # MPS does not guarantee bitwise reproducibility, even with this request.
    torch.use_deterministic_algorithms(True, warn_only=True)

def environment(device):
    import torchvision
    return {
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "device": str(device), "precision": "float32",
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "cpu_threads": torch.get_num_threads(), "pid": os.getpid(),
        "bitwise_determinism_guaranteed": False,
    }

def source_manifest():
    paths = [ROOT / "study.py", *sorted((ROOT / "poolstudy").glob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def freeze_packages(out):
    proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    (Path(out) / "environment-freeze.txt").write_text(proc.stdout or proc.stderr, encoding="utf-8")
