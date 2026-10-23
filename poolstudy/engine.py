from __future__ import annotations
import gc
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .common import (experiment_dir, lock_protocol, seed_all, sync, environment, write_json,
                     read_json, source_manifest, freeze_packages, fingerprint)
from .data import load_raw, make_split, Images, loader, conditions
from .models import Classifier, normalized_entropy, count_macs
from .metrics import classification_metrics

def run_dir(cfg, method, seed):
    return experiment_dir(cfg) / "runs" / f"{method}_seed{seed}"

def ensure_source(out):
    path = Path(out) / "source_sha256.json"
    current = source_manifest()
    if path.exists() and read_json(path) != current:
        raise RuntimeError("Source changed since experiment began. Use a new config name to avoid mixing implementations.")
    if not path.exists():
        write_json(path, current)

def save_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)

def load_checkpoint(path):
    # Only load this project's own checkpoints. Pickle is not safe for untrusted files.
    return torch.load(path, map_location="cpu", weights_only=False)

def learning_rate(cfg, epoch):
    warm = min(cfg["warmup_epochs"], cfg["epochs"])
    if epoch < warm:
        return cfg["lr"] * (epoch+1) / max(1,warm)
    progress = (epoch-warm) / max(1,cfg["epochs"]-warm-1)
    return cfg["lr"] * .5 * (1+math.cos(math.pi*progress))

@torch.inference_mode()
def clean_validation(model, dl, device):
    model.eval()
    loss, correct, total = 0., 0, 0
    for x, y, _, _ in dl:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss += float(F.cross_entropy(logits, y, reduction="sum").cpu())
        correct += int((logits.argmax(1)==y).sum().cpu())
        total += len(y)
    return loss/total, 100*correct/total

def train_one(cfg, method, seed, device, raw):
    out = lock_protocol(cfg)
    ensure_source(out)
    folder = run_dir(cfg, method, seed)
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = Classifier(cfg, method, seed).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                weight_decay=cfg["weight_decay"], nesterov=True)
    checkpoint = folder / "last.pt"
    history, start = [], 0
    if checkpoint.exists():
        state = load_checkpoint(checkpoint)
        if state["protocol_hash"] != fingerprint(cfg):
            raise RuntimeError("Checkpoint/config mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, start = state["history"], state["epoch"]+1
        if start >= cfg["epochs"]:
            # Recover a crash after checkpoint commit but before CSV/flag commit.
            pd.DataFrame(history).to_csv(folder/"history.csv", index=False)
            write_json(folder/"training_complete.json", {"epoch":cfg["epochs"], "selection":"fixed_final_epoch",
                       "synthetic":cfg["synthetic"], "protocol_hash":fingerprint(cfg)})
            print(f"SKIP completed training: {method} seed={seed}", flush=True)
            return
        print(f"RESUME {method} seed={seed} at epoch {start+1}", flush=True)
    else:
        write_json(folder/"environment.json", environment(device))
    xraw, yraw, _ = raw
    train_ids, valid_ids = make_split(yraw, cfg, out)
    vds = Images(xraw, yraw, valid_ids, cfg, split="validation")
    vloader = loader(vds, cfg)
    for epoch in range(start, cfg["epochs"]):
        # Epoch-derived RNG makes an epoch restart identical in data order and
        # augmentation, independent of the previous method's execution.
        seed_all(seed * 100003 + epoch)
        tds = Images(xraw, yraw, train_ids, cfg, train=True, seed=seed, epoch=epoch)
        dl = loader(tds, cfg, train=True, epoch=epoch, seed=seed)
        model.train()
        lr = learning_rate(cfg, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals = torch.zeros(5, device=device)
        n, attention_n = 0, 0
        sync(device)
        begin = time.perf_counter()
        for xb, yb, _, _ in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, penalty, weights = model(xb, diagnostics=True)
            ce = F.cross_entropy(logits, yb, label_smoothing=cfg["label_smoothing"])
            loss = ce + penalty
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise FloatingPointError(f"Non-finite training loss: {method}, seed={seed}, epoch={epoch+1}")
            loss.backward()
            optimizer.step()
            bs = len(yb)
            totals[0] += ce.detach()*bs
            totals[1] += penalty.detach()*bs
            totals[2] += (logits.detach().argmax(1)==yb).sum()
            if weights is not None:
                h = normalized_entropy(weights.detach())
                totals[3] += h.sum()
                totals[4] += (h < cfg["entropy_floor"]).sum()
                attention_n += bs
            n += bs
        sync(device)
        train_seconds = time.perf_counter()-begin
        values = totals.cpu().numpy()
        val_loss, val_acc = clean_validation(model, vloader, device)
        row = {"epoch":epoch+1, "lr":lr, "train_ce":float(values[0]/n),
            "train_penalty":float(values[1]/n), "train_accuracy":float(100*values[2]/n),
            "validation_loss":val_loss, "validation_accuracy":val_acc, "train_seconds":train_seconds,
            "train_entropy":float(values[3]/attention_n) if attention_n else None,
            "fraction_below_floor":float(values[4]/attention_n) if attention_n else None,
            "gem_p":float(model.pool.p.detach().cpu()) if method=="gem" else None}
        history.append(row)
        save_checkpoint(checkpoint, {"epoch":epoch, "model":model.state_dict(), "optimizer":optimizer.state_dict(),
             "history":history, "protocol_hash":fingerprint(cfg), "method":method, "seed":seed})
        # Checkpoint is the authoritative history after a crash between these writes.
        pd.DataFrame(history).to_csv(folder/"history.csv", index=False)
        print(f"{cfg['name']} {method} s{seed} [{epoch+1:03}/{cfg['epochs']}] "
              f"CE={row['train_ce']:.4f} val={val_acc:.2f}% train={train_seconds:.1f}s", flush=True)
    write_json(folder/"training_complete.json", {"epoch":cfg["epochs"], "selection":"fixed_final_epoch",
              "synthetic":cfg["synthetic"], "protocol_hash":fingerprint(cfg)})
    del model, optimizer
    gc.collect()
    if device.type=="mps":
        torch.mps.empty_cache()

def train_all(cfg, device):
    out = lock_protocol(cfg)
    ensure_source(out)
    freeze_packages(out)
    raw = load_raw(cfg, train=True, download=True)
    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            train_one(cfg, method, seed, device, raw)

@torch.inference_mode()
def predict(model, dl, device):
    model.eval()
    pieces = {k:[] for k in ("probs", "labels", "ids", "entropy", "effective_positions", "peak", "occluded_mass", "actual_area")}
    for x, y, ids, mask in dl:
        logits, _, weights = model(x.to(device), diagnostics=True)
        probs = logits.softmax(-1).cpu().numpy()
        bs = len(y)
        pieces["probs"].append(probs)
        pieces["labels"].append(y.numpy())
        pieces["ids"].append(ids.numpy())
        pieces["actual_area"].append(mask.mean((1,2)).numpy())
        if weights is None:
            for key in ("entropy", "effective_positions", "peak", "occluded_mass"):
                pieces[key].append(np.full(bs, np.nan, dtype=np.float32))
        else:
            w = weights.cpu()
            h = normalized_entropy(w)
            # Fractional input-mask coverage of each 4x4 feature cell. This is
            # a geometric overlap, not a receptive-field or explanation metric.
            small = F.avg_pool2d(mask[:,None], 4).flatten(1)
            pieces["entropy"].append(h.numpy())
            pieces["effective_positions"].append((h * math.log(w.shape[1])).exp().numpy())
            pieces["peak"].append(w.max(-1).values.numpy())
            pieces["occluded_mass"].append((w*small).sum(-1).numpy())
    return {k:np.concatenate(v) for k,v in pieces.items()}

def profile_model(model, cfg, device, folder):
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in (1, cfg["batch_size"]):
            x = torch.randn(batch,3,32,32,device=device)
            for _ in range(cfg["profile_warmup"]):
                model(x)
            sync(device)
            for trial in range(cfg["profile_trials"]):
                timings = []
                for _ in range(cfg["profile_steps"]):
                    sync(device)
                    t0 = time.perf_counter()
                    model(x)
                    sync(device)
                    timings.append((time.perf_counter()-t0)*1000)
                for iteration, ms in enumerate(timings):
                    rows.append({"batch_size":batch, "trial":trial, "iteration":iteration,
                        "ms_per_batch":ms, "amortized_ms_per_image":ms/batch})
    pd.DataFrame(rows).to_csv(folder/"latency_raw.csv", index=False)
    macs = count_macs(model, device)
    write_json(folder/"complexity.json", {"parameters":sum(p.numel() for p in model.parameters()),
        "supported_module_macs":macs, "supported_module_flops":2*macs,
        "operation_convention":"Conv2d and Linear only; excludes functional pooling, BN, activations, softmax, reductions.",
        "device":str(device), "precision":"FP32", "timing":"synchronized per forward; excludes data loading and host/device transfer"})

def evaluate_one(cfg, method, seed, device, raw):
    folder = run_dir(cfg, method, seed)
    result_dir = folder/"test"
    result_dir.mkdir(parents=True, exist_ok=True)
    conds = conditions(cfg)
    complete = result_dir/"evaluation_complete.json"
    if complete.exists() and all((result_dir/f"{c['name']}.npz").exists() for c in conds):
        print(f"SKIP completed evaluation: {method} seed={seed}", flush=True)
        return
    state = load_checkpoint(folder/"last.pt")
    if state["epoch"]+1 != cfg["epochs"] or state["protocol_hash"] != fingerprint(cfg):
        raise RuntimeError("Final checkpoint does not match evaluation protocol")
    model = Classifier(cfg, method, seed).to(device)
    model.load_state_dict(state["model"])
    x, y, classes = raw
    ids = np.arange(len(y))
    rows = []
    for cond in conds:
        target = result_dir/f"{cond['name']}.npz"
        if target.exists():
            with np.load(target) as archive:
                pred = {k:archive[k] for k in archive.files}
        else:
            ds = Images(x,y,ids,cfg,condition=cond,split="test")
            pred = predict(model, loader(ds,cfg), device)
            tmp = target.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **pred)
            tmp.replace(target)
        scores, reliability = classification_metrics(pred["probs"], pred["labels"], cfg["ece_bins"])
        scalar = {"method":method,"seed":seed,**cond,**scores,"n":len(y),
                  "actual_area":float(pred["actual_area"].mean())}
        for key in ("entropy", "effective_positions", "peak", "occluded_mass"):
            values = pred[key]
            scalar[key] = float(np.nanmean(values)) if np.isfinite(values).any() else None
        rows.append(scalar)
        df = pd.DataFrame({"original_id":pred["ids"], "label":pred["labels"],
            "prediction":pred["probs"].argmax(1), "confidence":pred["probs"].max(1),
            "correct":pred["probs"].argmax(1)==pred["labels"],
            **{k:pred[k] for k in ("entropy","effective_positions","peak","occluded_mass","actual_area")}})
        df.to_csv(result_dir/f"{cond['name']}_predictions.csv.gz", index=False)
        pd.DataFrame(reliability).to_csv(result_dir/f"{cond['name']}_reliability.csv", index=False)
        if cond["kind"]=="clean":
            predicted = pred["probs"].argmax(1)
            confusion = np.zeros((len(classes),len(classes)), dtype=int)
            np.add.at(confusion,(y,predicted),1)
            pd.DataFrame(confusion,index=classes,columns=classes).to_csv(result_dir/"confusion_clean.csv")
            counts = np.bincount(y,minlength=len(classes))
            pd.DataFrame({"class":classes,"n":counts,
                "accuracy":100*np.diag(confusion)/np.maximum(1,counts)}).to_csv(result_dir/"per_class_clean.csv",index=False)
        print(f"EVAL {method} s{seed} {cond['name']}: {scores['accuracy']:.2f}%", flush=True)
    pd.DataFrame(rows).to_csv(result_dir/"metrics.csv",index=False)
    # Profile one representative seed; repeated seeds are not independent devices.
    if seed==cfg["seeds"][0]:
        seed_all(seed)
        profile_model(model,cfg,device,folder)
    write_json(complete,{"protocol_hash":fingerprint(cfg),"synthetic":cfg["synthetic"],
               "conditions":len(conds),"images_per_condition":len(y),"source":source_manifest()})
    del model
    gc.collect()
    if device.type=="mps":
        torch.mps.empty_cache()

def evaluate_all(cfg, device):
    out = lock_protocol(cfg)
    ensure_source(out)
    # Reject an incomplete suite before loading any test labels.
    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            path = run_dir(cfg,method,seed)/"last.pt"
            if not path.exists() or load_checkpoint(path)["epoch"]+1 != cfg["epochs"]:
                raise RuntimeError(f"Finish all training before test evaluation: {method}, seed={seed}")
    write_json(out/"test_protocol.json", {"conditions":conditions(cfg),"selection":"fixed_final_epoch",
               "primary_endpoint":"mean_occluded_accuracy", "secondary_endpoints":["clean_accuracy","gaussian_accuracy","nll","ece"],
               "inferential_unit":"training seed; masks and images are repeated measurements, not independent runs"})
    raw = load_raw(cfg,train=False,download=True)
    write_json(out/"classes.json",raw[2])
    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            evaluate_one(cfg,method,seed,device,raw)

def benchmark(cfg,device,batches=20):
    """Measured full-epoch timing with real train and validation loaders.

    This deliberately uses real dataset I/O and augmentations, unlike a
    synthetic compute-only speed check. It never saves trained checkpoints.
    """
    raw = load_raw(cfg,train=True,download=True)
    train_ids, valid_ids = make_split(raw[1],cfg)
    rows=[]
    for method in cfg["methods"]:
        seed_all(cfg["seeds"][0])
        model=Classifier(cfg,method,cfg["seeds"][0]).to(device)
        opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        ds=Images(raw[0],raw[1],train_ids,cfg,train=True,seed=13,epoch=0)
        dl=loader(ds,cfg,train=True)
        times=[]
        it=iter(dl)
        for step in range(min(batches+5,len(dl))):
            sync(device)
            start=time.perf_counter()
            xb,yb,_,_=next(it)
            opt.zero_grad(set_to_none=True)
            logits,penalty=model(xb.to(device))
            (F.cross_entropy(logits,yb.to(device))+penalty).backward()
            opt.step()
            sync(device)
            if step>=5:
                times.append(time.perf_counter()-start)
        if not times:
            raise ValueError("Need >5 train batches for timing")
        vloader=loader(Images(raw[0],raw[1],valid_ids,cfg,split="validation"),cfg)
        sync(device)
        start=time.perf_counter()
        clean_validation(model,vloader,device)
        sync(device)
        vs=time.perf_counter()-start
        seconds=float(np.mean(times))*len(dl)+vs
        rows.append({"method":method,"train_seconds_per_batch":float(np.mean(times)),
            "batches_per_epoch":len(dl),"validation_seconds":vs,"estimated_epoch_seconds":seconds,
            "estimated_training_hours_all_seeds":seconds*cfg["epochs"]*len(cfg["seeds"])/3600})
        print(rows[-1],flush=True)
        del model,opt
        gc.collect()
    out=experiment_dir(cfg).parent/f"{cfg['name']}_timing"
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"budget.csv",index=False)
    write_json(out/"environment.json",environment(device))
    write_json(out/"config.json",cfg)
    hours=sum(r["estimated_training_hours_all_seeds"] for r in rows)
    print(f"Estimated training+validation: {hours:.2f} hours. Excludes final corruption evaluation, checkpoints, and sustained thermal slowdown.",flush=True)
    print(f"Timing saved: {out}",flush=True)
