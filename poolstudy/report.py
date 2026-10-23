from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import torch
from torch.nn import functional as F
from .common import experiment_dir, read_json, write_json, fingerprint
from .data import conditions, load_raw, Images, loader, STATS
from .models import Classifier
from .engine import run_dir, load_checkpoint, ensure_source
from .metrics import mean_interval, paired_test, holm_adjust

LABELS = {"gap":"Average", "max":"Max", "gem":"GeM", "attention":"Attention",
          "attention_entropy":"Attention + entropy", "attention_floor":"Attention + entropy floor",
          "mixed_attention":"Fixed Average + Attention"}
COLORS = dict(zip(LABELS, ["#536B89","#8C8C8C","#D19B38","#496CB2","#5B9E84","#B44778","#9075B0"]))

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.spines.top":False,
        "axes.spines.right":False, "figure.dpi":120, "savefig.dpi":300, "pdf.fonttype":42,
        "ps.fonttype":42, "axes.grid":True, "grid.alpha":.18, "axes.axisbelow":True})

def save_figure(fig, folder, name, synthetic):
    if synthetic:
        fig.suptitle("SYNTHETIC ENGINEERING CHECK — NOT RESEARCH RESULTS",fontsize=9,color="firebrick")
    bottom=.035 if name.startswith("pooling_weights_") else 0
    right=.92 if name.startswith("pooling_weights_") else 1
    fig.tight_layout(rect=(0,bottom,right,.96 if synthetic else 1))
    if hasattr(fig,"_pool_colorbar"):
        cax=fig.add_axes([.94,.35,.015,.25])
        fig.colorbar(fig._pool_colorbar,cax=cax,label="Spatial weight")
    fig.savefig(folder/f"{name}.png",bbox_inches="tight")
    fig.savefig(folder/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)

def aggregate(cfg):
    out=experiment_dir(cfg)
    ensure_source(out)
    tables,figures=out/"tables",out/"figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    all_metrics=[]; endpoints=[]; correlations=[]
    conds=conditions(cfg)
    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            folder=run_dir(cfg,method,seed)
            flag=folder/"test"/"evaluation_complete.json"
            if not flag.exists() or read_json(flag)["protocol_hash"] != fingerprint(cfg):
                raise RuntimeError(f"Missing completed evaluation for {method}, seed {seed}")
            frame=pd.read_csv(folder/"test"/"metrics.csv")
            if len(frame) != len(conds):
                raise RuntimeError("Incomplete condition metrics")
            all_metrics.append(frame)
            clean=frame.loc[frame.kind=="clean"].iloc[0]
            occ=frame.loc[frame.kind=="occlusion"]
            noise=frame.loc[frame.kind=="gaussian"]
            with np.load(folder/"test"/"clean.npz") as a:
                clean_correct=a["probs"].argmax(1)==a["labels"]
                entropy=a["entropy"].copy()
                original_ids=a["ids"].copy()
                confidence=a["probs"].max(1)
            per_image=[]
            for cond in conds:
                if cond["kind"]!="occlusion":
                    continue
                with np.load(folder/"test"/f"{cond['name']}.npz") as a:
                    if not np.array_equal(a["ids"],original_ids):
                        raise RuntimeError("Misaligned paired predictions")
                    per_image.append(a["probs"].argmax(1)==a["labels"])
            robust=np.mean(per_image,axis=0) if per_image else np.full(len(entropy),np.nan)
            conditional=float(100*robust[clean_correct].mean()) if clean_correct.any() and per_image else None
            curve=pd.concat([pd.DataFrame({"actual_area":[0.],"accuracy":[clean.accuracy]}),
                 occ.groupby("actual_area",as_index=False).accuracy.mean()]).sort_values("actual_area")
            auc=float(np.trapezoid(curve.accuracy,curve.actual_area)/curve.actual_area.max()) if len(curve)>1 else None
            row={"method":method,"seed":seed,"clean_accuracy":float(clean.accuracy),
                "mean_occluded_accuracy":float(occ.accuracy.mean()) if len(occ) else None,
                "occlusion_drop_pp":float(clean.accuracy-occ.accuracy.mean()) if len(occ) else None,
                "conditional_retention":conditional,"normalized_occlusion_auc":auc,
                "gaussian_accuracy":float(noise.accuracy.mean()) if len(noise) else None,
                "clean_nll":float(clean.nll),"clean_ece":float(clean.ece),"clean_brier":float(clean.brier),
                "mean_occluded_nll":float(occ.nll.mean()) if len(occ) else None,
                "mean_occluded_ece":float(occ.ece.mean()) if len(occ) else None,
                "clean_entropy":float(clean.entropy) if pd.notna(clean.entropy) else None}
            endpoints.append(row)
            if np.isfinite(entropy).all() and np.std(entropy)>1e-8 and per_image:
                rho=float(stats.spearmanr(entropy,robust).statistic) if np.std(robust)>0 else None
                subset=clean_correct
                conditional_rho=float(stats.spearmanr(entropy[subset],robust[subset]).statistic) if (
                    subset.sum()>2 and np.std(entropy[subset])>1e-8 and np.std(robust[subset])>0) else None
                correlations.append({"method":method,"seed":seed,"spearman_entropy_vs_robustness":rho,
                   "spearman_among_clean_correct":conditional_rho,"n_images":len(entropy),
                   "interpretation":"descriptive within-run association; confounded by class and difficulty; no causal claim"})
                pd.DataFrame({"original_id":original_ids,"clean_entropy":entropy,"clean_confidence":confidence,
                    "clean_correct":clean_correct,"fraction_occluded_correct":robust}).to_csv(
                        tables/f"association_{method}_s{seed}.csv.gz",index=False)
    metrics=pd.concat(all_metrics,ignore_index=True)
    seed_df=pd.DataFrame(endpoints)
    metrics.to_csv(tables/"all_condition_metrics.csv",index=False)
    seed_df.to_csv(tables/"seed_endpoints.csv",index=False)
    pd.DataFrame(correlations).to_csv(tables/"entropy_robustness_correlations.csv",index=False)
    summaries=[]
    for method, group in seed_df.groupby("method",sort=False):
        for metric in seed_df.columns.drop(["method","seed"]):
            values=group[metric].dropna().to_numpy()
            if len(values):
                summaries.append({"method":method,"metric":metric,**mean_interval(values)})
    summary=pd.DataFrame(summaries)
    summary.to_csv(tables/"summary_mean_sd_ci.csv",index=False)
    tests=[]
    # Three predeclared primary comparisons form one Holm family. Clean
    # accuracy is a separate secondary family, never a selected winner metric.
    target="attention_floor"
    comparators=[m for m in ("attention","attention_entropy","mixed_attention") if m in cfg["methods"]]
    if target in cfg["methods"]:
        for metric in ("mean_occluded_accuracy","clean_accuracy"):
            family=[]
            for comparator in comparators:
                a=seed_df[seed_df.method==target].set_index("seed")[metric]
                b=seed_df[seed_df.method==comparator].set_index("seed")[metric]
                matched=pd.concat([a,b],axis=1,keys=["a","b"]).dropna()
                if matched.empty:
                    continue
                differences=matched.a-matched.b
                family.append({"target":target,"comparator":comparator,"metric":metric,
                    "seeds":";".join(map(str,matched.index)),**paired_test(differences)})
            valid=[r for r in family if r["p_raw"] is not None]
            adjusted=holm_adjust([r["p_raw"] for r in valid])
            for row,p in zip(valid,adjusted):
                row["p_holm"]=float(p)
            tests.extend(family)
    pd.DataFrame(tests).to_csv(tables/"paired_seed_tests.csv",index=False)
    # A compact reviewer-friendly table with unmodified means and seed SDs.
    compact=[]
    for method in cfg["methods"]:
        row={"Method":LABELS[method]}
        for key,label in (("clean_accuracy","Clean accuracy (%)"),("mean_occluded_accuracy","Occluded accuracy (%)"),
                          ("occlusion_drop_pp","Drop (pp)"),("clean_nll","Clean NLL"),("clean_ece","Clean ECE")):
            s=summary[(summary.method==method)&(summary.metric==key)]
            if len(s):
                r=s.iloc[0]
                row[label]=f"{r['mean']:.3f}"+(f" ± {r.sd:.3f}" if pd.notna(r.sd) else " (one seed)")
        compact.append(row)
    compact=pd.DataFrame(compact)
    compact.to_csv(tables/"paper_table.csv",index=False)
    (tables/"paper_table.tex").write_text(compact.to_latex(index=False,escape=True),encoding="utf-8")
    complexity=[]; latency=[]
    for method in cfg["methods"]:
        folder=run_dir(cfg,method,cfg["seeds"][0])
        if (folder/"complexity.json").exists():
            comp=read_json(folder/"complexity.json")
            complexity.append({"method":method,**comp})
        if (folder/"latency_raw.csv").exists():
            lat=pd.read_csv(folder/"latency_raw.csv")
            # Dispersion is across trial means on one device, not seed SD.
            for batch,g in lat.groupby("batch_size"):
                trials=g.groupby("trial").ms_per_batch.mean()
                latency.append({"method":method,"batch_size":batch,**mean_interval(trials),
                                "unit":"ms per batch; synchronized forward only"})
    pd.DataFrame(complexity).to_csv(tables/"complexity.csv",index=False)
    pd.DataFrame(latency).to_csv(tables/"latency_summary.csv",index=False)
    return metrics,seed_df,summary

def plots(cfg, metrics, seed_df, summary):
    style()
    out=experiment_dir(cfg); figures=out/"figures"
    # Exact architecture diagram from the configuration, without generated art.
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig,ax=plt.subplots(figsize=(11,3.6))
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    channels=cfg["width"]*4
    classes=10 if cfg["dataset"]=="cifar10" else 100
    boxes=[(.02,.62,"Input\n3 × 32 × 32"),(.22,.62,f"{cfg['backbone'].replace('resnet','ResNet-')}\n{channels} × 8 × 8"),
           (.42,.62,"1 × 1 spatial score\nsoftmax → a"),(.62,.62,f"Weighted sum\nz ∈ R^{channels}"),
           (.82,.62,f"Linear classifier\n{classes} logits")]
    for x,y,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),.16,.22,boxstyle="round,pad=0.008",facecolor="#F4F5F8",edgecolor="#536B89"))
        ax.text(x+.08,y+.11,label,ha="center",va="center",fontsize=8.5)
    for start,end in [((.18,.73),(.22,.73)),((.38,.73),(.42,.73)),((.58,.73),(.62,.73)),((.78,.73),(.82,.73)),
                      ((.50,.62),(.50,.40)),((.59,.29),(.69,.29)),((.90,.62),(.90,.40))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="->",mutation_scale=12,color="#536B89"))
    ax.add_patch(FancyBboxPatch((.30,.18),.29,.22,boxstyle="round,pad=.008",facecolor="#FBEFF5",edgecolor="#B44778"))
    ax.text(.445,.29,f"h = H(a) / log(64)\nPenalty = max(0, {cfg['entropy_floor']:g} − h)²",ha="center",va="center",fontsize=8.5)
    ax.add_patch(FancyBboxPatch((.69,.18),.29,.22,boxstyle="round,pad=.008",facecolor="#FBEFF5",edgecolor="#B44778"))
    ax.text(.835,.29,f"Training loss\nCE + {cfg['floor_lambda']:g} × mean(Penalty)",ha="center",va="center",fontsize=8.5)
    ax.text(.50,.06,"Entropy penalty is used only during training. All pooling methods share the same backbone and classifier initialization.",ha="center",fontsize=8)
    save_figure(fig,figures,"architecture_entropy_floor",cfg["synthetic"])
    # Clean validation curves, with variation across training seeds.
    fig,axs=plt.subplots(1,2,figsize=(10,3.4))
    for method in cfg["methods"]:
        frames=[]
        for seed in cfg["seeds"]:
            f=pd.read_csv(run_dir(cfg,method,seed)/"history.csv")
            f["seed"]=seed; frames.append(f)
        all_h=pd.concat(frames)
        for ax,key in zip(axs,("validation_accuracy","train_ce")):
            grouped=all_h.groupby("epoch")[key]
            mean=grouped.mean(); sd=grouped.std().fillna(0)
            ax.plot(mean.index,mean,label=LABELS[method],color=COLORS[method])
            ax.fill_between(mean.index,mean-sd,mean+sd,color=COLORS[method],alpha=.1)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy (%)" if key=="validation_accuracy" else "Training CE")
    axs[0].legend(fontsize=6,loc="lower right")
    save_figure(fig,figures,"learning_curves",cfg["synthetic"])
    fills=cfg["fills"]
    fig,axs=plt.subplots(1,max(1,len(fills)),figsize=(5*max(1,len(fills)),3.6),squeeze=False)
    for ax,fill in zip(axs[0],fills):
        for method in cfg["methods"]:
            frame=metrics[(metrics.method==method)&((metrics.fill==fill)|(metrics.kind=="clean"))]
            byseed=frame.groupby(["seed","actual_area"],as_index=False).accuracy.mean()
            g=byseed.groupby("actual_area").accuracy
            mean=g.mean(); sd=g.std().fillna(0)
            ax.errorbar(mean.index*100,mean,yerr=sd,marker="o",ms=3,capsize=2,label=LABELS[method],color=COLORS[method])
        ax.set(xlabel="Actual occluded pixel area (%)",ylabel="Accuracy (%)",title=f"{fill.capitalize()} filled square")
        ax.legend(fontsize=6)
    save_figure(fig,figures,"occlusion_curves",cfg["synthetic"])
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    methods=cfg["methods"]
    for ax,key in zip(axs,("clean_accuracy","mean_occluded_accuracy")):
        vals=seed_df.groupby("method")[key]
        means=vals.mean().reindex(methods); sds=vals.std().reindex(methods).fillna(0)
        ax.bar(np.arange(len(methods)),means,yerr=sds,capsize=3,color=[COLORS[m] for m in methods],alpha=.85)
        for i,m in enumerate(methods):
            points=seed_df[seed_df.method==m][key]
            ax.scatter(np.full(len(points),i),points,color="black",s=10,zorder=3)
        ax.set_xticks(np.arange(len(methods)),[LABELS[m] for m in methods],rotation=35,ha="right",fontsize=7)
        ax.set_ylabel("Accuracy (%)"); ax.set_title("Clean" if key=="clean_accuracy" else "Mean across all occlusion conditions")
    save_figure(fig,figures,"accuracy_comparison",cfg["synthetic"])
    fig,ax=plt.subplots(figsize=(6,3.8))
    for method in methods:
        frame=metrics[(metrics.method==method)&metrics.entropy.notna()]
        frame=frame[frame.kind.isin(["clean","occlusion"])]
        byseed=frame.groupby(["seed","actual_area"],as_index=False).entropy.mean()
        if len(byseed):
            g=byseed.groupby("actual_area").entropy
            mean=g.mean(); sd=g.std().fillna(0)
            ax.errorbar(mean.index*100,mean,yerr=sd,marker="o",capsize=2,label=LABELS[method],color=COLORS[method])
    ax.axhline(cfg["entropy_floor"],ls="--",color="gray",label="Training entropy floor")
    ax.set(xlabel="Actual occluded pixel area (%)",ylabel="Normalized spatial entropy",ylim=(0,1.05))
    ax.legend(fontsize=7)
    save_figure(fig,figures,"spatial_entropy",cfg["synthetic"])
    # Reliability on the clean set, pooled descriptively over model runs.
    fig,ax=plt.subplots(figsize=(5,4))
    ax.plot([0,1],[0,1],"--",color="gray")
    for method in methods:
        frames=[pd.read_csv(run_dir(cfg,method,s)/"test"/"clean_reliability.csv") for s in cfg["seeds"]]
        f=pd.concat(frames).dropna()
        f["correct_mass"]=f.accuracy*f["count"]; f["confidence_mass"]=f.confidence*f["count"]
        g=f.groupby("bin")[["correct_mass","confidence_mass","count"]].sum()
        ax.plot(g.confidence_mass/g["count"],g.correct_mass/g["count"],marker="o",ms=3,label=LABELS[method],color=COLORS[method])
    ax.set(xlabel="Mean confidence",ylabel="Empirical accuracy",xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=6)
    save_figure(fig,figures,"reliability_clean",cfg["synthetic"])
    # Show paired seed-level effects with intervals, including zero.
    path=out/"tables"/"paired_seed_tests.csv"
    if path.exists() and path.stat().st_size>2:
        tests=pd.read_csv(path)
        tests=tests[tests.metric=="mean_occluded_accuracy"]
        if len(tests):
            fig,ax=plt.subplots(figsize=(7,2.8))
            for i,r in enumerate(tests.itertuples()):
                err=np.array([[r.mean-r.ci_low],[r.ci_high-r.mean]]) if pd.notna(r.ci_low) else None
                ax.errorbar(r.mean,i,xerr=err,fmt="o",capsize=4,color=COLORS["attention_floor"])
            ax.axvline(0,color="gray",ls="--")
            ax.set_yticks(range(len(tests)),["Floor minus "+LABELS[m] for m in tests.comparator])
            ax.set_xlabel("Paired occlusion accuracy difference (pp), 95% t interval")
            save_figure(fig,figures,"paired_effects",cfg["synthetic"])
    # Training diagnostics reveal whether the proposed penalty actually acted.
    fig,axs=plt.subplots(1,2,figsize=(10,3.3))
    for method in [m for m in methods if m.startswith("attention")]:
        frames=[pd.read_csv(run_dir(cfg,method,s)/"history.csv") for s in cfg["seeds"]]
        f=pd.concat(frames)
        for ax,key in zip(axs,("train_entropy","fraction_below_floor")):
            g=f.groupby("epoch")[key].mean()
            ax.plot(g.index,g,label=LABELS[method],color=COLORS[method])
            ax.set_xlabel("Epoch")
    axs[0].axhline(cfg["entropy_floor"],color="gray",ls="--")
    axs[0].set_ylabel("Training normalized entropy")
    axs[1].set_ylabel("Fraction of samples below entropy floor")
    axs[1].legend(fontsize=7)
    save_figure(fig,figures,"training_entropy_diagnostics",cfg["synthetic"])
    fig,ax=plt.subplots(figsize=(6,3.5))
    for method in [m for m in methods if m.startswith("attention") or m=="mixed_attention"]:
        arrays=[]
        for seed in cfg["seeds"]:
            with np.load(run_dir(cfg,method,seed)/"test"/"clean.npz") as a:
                arrays.append(a["entropy"])
        ax.hist(np.concatenate(arrays),bins=np.linspace(0,1,41),density=True,histtype="step",
                linewidth=1.4,label=LABELS[method],color=COLORS[method])
    ax.set(xlabel="Normalized spatial entropy",ylabel="Density (samples pooled descriptively across seeds)")
    ax.legend(fontsize=7)
    save_figure(fig,figures,"entropy_distribution_clean",cfg["synthetic"])
    if (metrics.kind=="gaussian").any():
        fig,ax=plt.subplots(figsize=(6,3.6))
        for method in methods:
            frame=metrics[(metrics.method==method)&metrics.kind.isin(["clean","gaussian"])]
            g=frame.groupby("sigma").accuracy
            mu=g.mean(); sd=g.std().fillna(0)
            ax.errorbar(mu.index,mu,yerr=sd,marker="o",capsize=2,color=COLORS[method],label=LABELS[method])
        ax.set(xlabel="Gaussian noise standard deviation in [0,1] pixel space",ylabel="Accuracy (%)")
        ax.legend(fontsize=6)
        save_figure(fig,figures,"gaussian_noise_secondary",cfg["synthetic"])
    for method in [m for m in ("attention","attention_floor") if m in methods]:
        counts=None
        for seed in cfg["seeds"]:
            matrix=pd.read_csv(run_dir(cfg,method,seed)/"test"/"confusion_clean.csv",index_col=0)
            counts=matrix.to_numpy() if counts is None else counts+matrix.to_numpy()
        normalized=counts/np.maximum(1,counts.sum(1,keepdims=True))
        fig,ax=plt.subplots(figsize=(5.2,4.4))
        im=ax.imshow(normalized,vmin=0,vmax=1,cmap="Blues")
        tick_step=1 if len(counts)<=10 else 10
        ax.set_xticks(range(0,len(counts),tick_step)); ax.set_yticks(range(0,len(counts),tick_step))
        ax.set(xlabel="Predicted class index",ylabel="True class index",title=LABELS[method]+" clean confusion")
        fig.colorbar(im,ax=ax,label="Row-normalized fraction")
        save_figure(fig,figures,f"confusion_{method}",cfg["synthetic"])

@torch.inference_mode()
def qualitative(cfg,device):
    style()
    out=experiment_dir(cfg); figures=out/"figures"
    raw=load_raw(cfg,train=False,download=False)
    ids=[i for i in cfg["qualitative_ids"] if i<len(raw[1])]
    methods=[m for m in cfg["methods"] if m in ("gap","attention","attention_entropy","attention_floor","mixed_attention")]
    conds=[conditions(cfg)[0]]
    for fill in cfg["fills"]:
        candidates=[c for c in conditions(cfg) if c["kind"]=="occlusion" and c["fill"]==fill and c["repeat"]==0]
        if candidates:
            conds.append(min(candidates,key=lambda c:abs(c["area"]-.25)))
    if not ids or not methods:
        return
    seed=cfg["seeds"][0]
    collected={}
    for method in methods:
        model=Classifier(cfg,method,seed).to(device)
        model.load_state_dict(load_checkpoint(run_dir(cfg,method,seed)/"last.pt")["model"])
        model.eval()
        for cond in conds:
            ds=Images(raw[0],raw[1],ids,cfg,condition=cond,split="test")
            batches=list(loader(ds,cfg))
            # The fixed qualitative IDs are small; concatenate even if eval batch size is smaller.
            xb=torch.cat([b[0] for b in batches]); yb=torch.cat([b[1] for b in batches])
            logits,_,w=model(xb.to(device),diagnostics=True)
            collected[method,cond["name"]]=(xb.cpu(),yb.cpu(),logits.argmax(1).cpu(),w.cpu())
        del model
    for cond in conds:
        fig,axs=plt.subplots(len(methods)+1,len(ids),figsize=(2.0*len(ids),1.8*(len(methods)+1)),squeeze=False)
        mean=torch.tensor(STATS[cfg["dataset"]][0])[:,None,None]
        std=torch.tensor(STATS[cfg["dataset"]][1])[:,None,None]
        for col,index in enumerate(ids):
            xb,yb,_,_=collected[methods[0],cond["name"]]
            im=(xb[col]*std+mean).permute(1,2,0).numpy().clip(0,1)
            axs[0,col].imshow(im); axs[0,col].set_title(f"ID {index}\n{raw[2][int(yb[col])]}",fontsize=8)
            for row,method in enumerate(methods,1):
                _,_,pred,w=collected[method,cond["name"]]
                heat=F.interpolate(w[col].reshape(1,1,8,8),size=(32,32),mode="bilinear",align_corners=False)[0,0].numpy()
                axs[row,col].imshow(im)
                # Fixed scale for all maps; saturation at 0.08 is disclosed.
                artist=axs[row,col].imshow(heat,cmap="magma",alpha=.55,vmin=0,vmax=.08)
                axs[row,col].set_title(f"{LABELS[method]}\nPred: {raw[2][int(pred[col])]}",fontsize=7)
        for ax in axs.flat:
            ax.axis("off")
        fig.text(.47,.008,"Spatial pooling weights, fixed range 0–0.08; interpolated from 8×8. Not class activation maps.",ha="center",fontsize=7)
        # Add the colorbar after layout to reserve the same region in every panel.
        # save_figure reserves the right margin for this standalone color axis.
        fig._pool_colorbar=artist
        save_figure(fig,figures,f"pooling_weights_{cond['name']}",cfg["synthetic"])
    # Save actual corrupted inputs separately for paper illustration.
    for cond in conds:
        xb,_,_,_=collected[methods[0],cond["name"]]
        image_dir=out/"images"/cond["name"]
        image_dir.mkdir(parents=True,exist_ok=True)
        for i,index in enumerate(ids):
            im=(xb[i]*std+mean).permute(1,2,0).numpy().clip(0,1)
            plt.imsave(image_dir/f"id_{index:05d}.png",im)

def report(cfg,device):
    metrics,seeds,summary=aggregate(cfg)
    plots(cfg,metrics,seeds,summary)
    qualitative(cfg,device)
    out=experiment_dir(cfg)
    status="SYNTHETIC CHECK ONLY. Not valid scientific evidence." if cfg["synthetic"] else "Measured experimental results. Interpret with seed uncertainty."
    text=f"""# Experiment report\n\n{status}\n\nDataset: {cfg['dataset']}; backbone: {cfg['backbone']}; seeds: {cfg['seeds']}; epochs: {cfg['epochs']}.\n\nThe primary endpoint is the unweighted mean accuracy over every specified occlusion fill, area, and mask repeat, first reduced within each training seed. Clean accuracy is secondary. Masks and test images are not independent training runs.\n\nSee tables/paper_table.csv and paired_seed_tests.csv. Three primary floor-versus-control tests use Holm adjustment. Intervals are unadjusted paired t intervals. Small seed counts have low power. Do not claim an improvement from the largest mean alone.\n\nUse figures/*.pdf for vector manuscript figures and *.png for previews. All qualitative examples use prespecified image IDs and the first seed; they are not selected successes. Attention weights are not causal explanations.\n\nThe entropy-floor loss is an exploratory soft penalty, not a guaranteed hard entropy constraint. A higher entropy does not establish robustness. If the penalty is inactive, the floor and ordinary-attention runs may coincide; inspect fraction_below_floor and train_penalty in history.csv.\n\nSynthetic occlusions are not CIFAR-C or real-world occlusion labels. Accuracy need not decline monotonically. Report actual_area, not just the requested fraction. GeM and Max do not define a single shared spatial probability map, so their entropy fields are intentionally missing.\n"""
    (out/"RESULTS_README.md").write_text(text,encoding="utf-8")
    print(f"Report created: {out}",flush=True)
