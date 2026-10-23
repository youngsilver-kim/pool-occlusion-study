#!/usr/bin/env python3
"""Optional sensitivity analysis on TRAIN-derived validation data only.

No official test image or label is loaded. This does not silently select or
modify the main protocol. All development models live in separate directories.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from poolstudy.common import read_config,choose_device,experiment_dir,write_json
from poolstudy.data import load_raw,make_split,Images,loader,conditions
from poolstudy.engine import train_all,run_dir,load_checkpoint,predict
from poolstudy.models import Classifier
from poolstudy.metrics import classification_metrics

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",default=str(ROOT/"configs"/"main.json"))
    ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--device",choices=["mps","cpu","cuda","auto"],default=None)
    args=ap.parse_args()
    base=read_config(args.config)
    if args.device:
        base["device"]=args.device
    device=choose_device(base["device"])
    torch.set_num_threads(base["cpu_threads"])
    if args.epochs<1:
        raise SystemExit("epochs must be positive")
    raw=load_raw(base,train=True,download=True)
    _,valid=make_split(raw[1],base)
    plans=[("attention_floor",{"entropy_floor":h,"floor_lambda":1.}) for h in (.6,.8,.9)]
    plans += [("attention_floor",{"entropy_floor":.8,"floor_lambda":v}) for v in (.2,5.)]
    plans += [("attention_entropy",{"entropy_lambda":v}) for v in (.01,.05,.2)]
    rows=[]
    for method,params in plans:
        cfg=copy.deepcopy(base)
        cfg.update(params)
        cfg.update(epochs=args.epochs,seeds=[base["seeds"][0]],methods=[method])
        suffix="_".join(f"{k}{v:g}" for k,v in params.items())
        cfg["name"]=f"development_{base['dataset']}_{base['backbone']}_e{args.epochs}_{method}_{suffix}"
        train_all(cfg,device)
        model=Classifier(cfg,method,cfg["seeds"][0]).to(device)
        model.load_state_dict(load_checkpoint(run_dir(cfg,method,cfg["seeds"][0])/"last.pt")["model"])
        for cond in conditions(cfg):
            ds=Images(raw[0],raw[1],valid,cfg,condition=cond,split="validation")
            pred=predict(model,loader(ds,cfg),device)
            scores,_=classification_metrics(pred["probs"],pred["labels"],cfg["ece_bins"])
            entropy=float(np.nanmean(pred["entropy"])) if np.isfinite(pred["entropy"]).any() else None
            rows.append({"development_run":cfg["name"],"method":method,"epochs":args.epochs,"seed":cfg["seeds"][0],
                 "entropy_floor":cfg["entropy_floor"],"floor_lambda":cfg["floor_lambda"],
                 "entropy_lambda":cfg["entropy_lambda"],**cond,**scores,"entropy":entropy,
                 "split":"train_derived_validation_only"})
        del model
    out=experiment_dir(base).parent/f"development_summary_{base['dataset']}_{base['backbone']}_e{args.epochs}"
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"sensitivity_validation.csv",index=False)
    reduced=frame.groupby(["development_run","method","kind"],as_index=False).accuracy.mean()
    reduced.to_csv(out/"sensitivity_summary.csv",index=False)
    occ=reduced[reduced.kind=="occlusion"]
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(range(len(occ)),occ.accuracy,color="#B44778")
    ax.set_xticks(range(len(occ)),[r.replace(f"development_{base['dataset']}_{base['backbone']}_e{args.epochs}_","") for r in occ.development_run],rotation=35,ha="right",fontsize=7)
    ax.set_ylabel("Validation mean occluded accuracy (%)")
    ax.set_title("SYNTHETIC CHECK ONLY — not research results" if base["synthetic"] else "Development sensitivity — one seed, validation only")
    fig.tight_layout(); fig.savefig(out/"sensitivity.png",dpi=300); fig.savefig(out/"sensitivity.pdf"); plt.close(fig)
    write_json(out/"notes.json",{"official_test_loaded":False,"automatic_model_selection":False,
        "epochs":args.epochs,"seed":base["seeds"][0],"base_config":base,
        "warning":"One-seed development sensitivity. If used for selection, freeze a new protocol before any official test evaluation."})
    print(f"Sensitivity complete: {out}")

if __name__=="__main__":
    main()
