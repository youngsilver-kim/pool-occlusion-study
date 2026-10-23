#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Keep all caches inside this project, never modify another research environment.
ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
os.environ.setdefault("TORCH_HOME",str(ROOT/".cache"/"torch"))

def main():
    parser=argparse.ArgumentParser(description="Controlled pooling/occlusion study. Read README_KO.md first.")
    parser.add_argument("command",choices=["doctor","download","benchmark","train","evaluate","report","all"])
    parser.add_argument("--config",default=str(ROOT/"configs"/"main.json"))
    parser.add_argument("--device",choices=["auto","mps","cuda","cpu"],default=None)
    parser.add_argument("--batches",type=int,default=20,help="Measured train batches after 5 warmup batches for benchmark")
    args=parser.parse_args()
    try:
        import torch
        from poolstudy.common import read_config,choose_device,environment,experiment_dir,write_json
        from poolstudy.data import load_raw
        from poolstudy.engine import train_all,evaluate_all,benchmark
        from poolstudy.report import report
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Run bash setup_macos.sh, then source .venv/bin/activate.")
    cfg=read_config(args.config)
    if args.device is not None:
        cfg["device"]=args.device
    torch.set_num_threads(cfg["cpu_threads"])
    device=choose_device(cfg["device"])
    print(f"Device={device}; mode={'SYNTHETIC CHECK ONLY' if cfg['synthetic'] else 'REAL DATA'}; config={cfg['name']}",flush=True)
    if args.command=="doctor":
        print(json.dumps(environment(device),indent=2,ensure_ascii=False))
        if device.type=="cpu":
            print("CPU selected. For M5 research runs, confirm native arm64 Python and MPS availability.")
        return
    if args.command=="download":
        for train in (True,False):
            raw=load_raw(cfg,train=train,download=True)
            print(f"{'Train' if train else 'Test'}: {len(raw[1])} images, {len(raw[2])} classes")
        return
    # One process owns a configuration's output. Recover stale locks after a killed process.
    out=experiment_dir(cfg)
    if args.command=="benchmark":
        benchmark(cfg,device,args.batches)
        return
    out.mkdir(parents=True,exist_ok=True)
    lock=out/"active.pid"
    if lock.exists():
        try:
            pid=int(lock.read_text())
            os.kill(pid,0)
        except (ProcessLookupError,ValueError):
            lock.unlink(missing_ok=True)
        else:
            raise SystemExit(f"Another process owns this configuration (PID {pid}).")
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try:
        if args.command in ("train","all"):
            train_all(cfg,device)
        if args.command in ("evaluate","all"):
            evaluate_all(cfg,device)
        if args.command in ("report","all"):
            report(cfg,device)
    finally:
        lock.unlink(missing_ok=True)

if __name__=="__main__":
    main()
