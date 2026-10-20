"""Validation-only XAI faithfulness audit. Not clinical lesion localization."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
from .core import build_model, load_checkpoint, device_choice, seed_all, write_json
from .data import parse_manifest, transform
from .explain import gradcam
from .xai_checks import (FRACTIONS, rank_regions, occlusion_importance, deletion_probabilities,
                         mean_integrated_drop, spearman_map_correlation, paired_bootstrap)

MEAN=torch.tensor([.485,.456,.406]).view(3,1,1)
STD=torch.tensor([.229,.224,.225]).view(3,1,1)
METHODS=("gradcam","random","occlusion")

def image_to_rgb(x):
    return (x.detach().cpu()*STD+MEAN).clamp(0,1).permute(1,2,0).numpy()

def image_tensor(path,size):
    with Image.open(path) as im:
        return transform(size)(im.convert("RGB"))

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def run(args):
    seed_all(args.seed)
    if not 2<=args.grid<=16:raise ValueError("grid must be in [2,16]")
    if args.per_grade<1:raise ValueError("per-grade must be >=1")
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    device=device_choice()
    ck=load_checkpoint(args.checkpoint,device)
    size=int(ck["image_size"])
    if size%args.grid:raise ValueError("Image size must divide the grid")
    model=build_model(ck["model_name"],pretrained=False).to(device)
    model.load_state_dict(ck["model_state"],strict=True);model.eval()
    val=parse_manifest(args.val_csv,args.images_root,filename_col="filename",label_col="grade")
    selected=pd.concat([part.sample(n=min(args.per_grade,len(part)),random_state=args.seed)
                        for _,part in val.groupby("grade")],ignore_index=True)
    assert selected["grade"].nunique()==5, "Must include every DR severity grade"
    print("Grade-balanced XAI sample:",len(selected),selected.grade.value_counts().sort_index().to_dict(),flush=True)
    results_file=out/"per_image.csv"
    df=pd.read_csv(results_file) if results_file.exists() else pd.DataFrame()
    completed=set(df["filename"].astype(str)) if len(df) else set()
    illustrations=[];pictured_grades=set();extra_failure=False
    mean=MEAN.to(device);std=STD.to(device)
    for i,row in selected.iterrows():
        filename=Path(row.path).name
        if filename in completed:continue
        x=image_tensor(row.path,size).to(device)
        with torch.no_grad():
            rgb=(x*std+mean).clamp(0,1)
            baseline=(gaussian_blur(rgb,kernel_size=[41,41],sigma=[12.,12.])-mean)/std
            probabilities=model(x.unsqueeze(0)).softmax(1)[0]
            target=int(probabilities.argmax().item());orig=float(probabilities[target].item())
        cam,target_cam,_=gradcam(model,x.detach().clone().unsqueeze(0),target_class=target)
        assert target_cam==target
        camgrid=F.adaptive_avg_pool2d(torch.tensor(cam,dtype=torch.float32)[None,None],
                                      (args.grid,args.grid))[0,0].numpy().ravel()
        occ=occlusion_importance(model,x,baseline,target,orig,args.grid,device,args.batch_size)
        ranks={
            "gradcam":rank_regions(camgrid),
            "occlusion":rank_regions(occ),
            "random":np.random.default_rng(args.seed*100_003+i).permutation(args.grid*args.grid),
        }
        curves={name:deletion_probabilities(model,x,baseline,rank,target,args.grid,device,args.batch_size)
                for name,rank in ranks.items()}
        result={"filename":filename,"true_grade":int(row.grade),"pred_grade":target,
                "confidence":orig,"correct":int(int(row.grade)==target),
                "undergraded_by":int(row.grade)-target,
                "gradcam_occlusion_spearman":spearman_map_correlation(camgrid,occ)}
        for name in METHODS:
            result[f"aop_drop_{name}"]=mean_integrated_drop(curves[name])
            for fraction,p in zip(FRACTIONS,curves[name]):
                result[f"p_{name}_{fraction:.3f}"]=float(p)
        df=pd.concat([df,pd.DataFrame([result])],ignore_index=True)
        df.to_csv(results_file,index=False)  # Per-image recovery from interruption
        if args.figures>0:
            grade=int(row.grade)
            add_grade=grade not in pictured_grades
            add_failure=grade>=3 and grade>target and not extra_failure
            if len(illustrations)<args.figures and (add_grade or add_failure):
                illustrations.append((filename,image_to_rgb(x),cam,occ.reshape(args.grid,args.grid),result))
                if add_grade:pictured_grades.add(grade)
                else:extra_failure=True
        if len(df)%5==0:print("Completed",len(df),"/",len(selected),flush=True)
    df=df.sort_values("filename").reset_index(drop=True)
    sanity_file=out/"classifier_randomization.csv"
    if not sanity_file.exists():
        subset=pd.concat([g.head(max(1,args.sanity_images//5))
                          for _,g in selected.groupby("grade")],ignore_index=True).head(args.sanity_images)
        originals={}
        for _,row in subset.iterrows():
            x=image_tensor(row.path,size).to(device)
            with torch.inference_mode():target=int(model(x[None]).argmax(1).item())
            original,_,_=gradcam(model,x.detach().clone().unsqueeze(0),target_class=target)
            gridmap=F.adaptive_avg_pool2d(torch.tensor(original)[None,None],
                                           (args.grid,args.grid))[0,0].numpy()
            originals[Path(row.path).name]=(x,target,gridmap)
        weight=model.fc.weight.detach().clone()
        bias=model.fc.bias.detach().clone()
        rows=[]
        try:
            torch.manual_seed(args.seed+777)
            model.fc.reset_parameters();model.eval()
            for filename,(x,target,original) in originals.items():
                changed,_,_=gradcam(model,x.detach().clone().unsqueeze(0),target_class=target)
                reduced=F.adaptive_avg_pool2d(torch.tensor(changed)[None,None],
                                              (args.grid,args.grid))[0,0].numpy()
                rows.append({"filename":filename,
                             "fc_randomized_gradcam_spearman":spearman_map_correlation(original,reduced),
                             "target_grade":target})
        finally:
            with torch.no_grad():
                model.fc.weight.copy_(weight);model.fc.bias.copy_(bias)
            model.eval()
        pd.DataFrame(rows).to_csv(sanity_file,index=False)
    sanity=pd.read_csv(sanity_file)
    if illustrations:
        folder=out/"illustrations";folder.mkdir(exist_ok=True)
        for k,(filename,rgb,cam,occ,case) in enumerate(illustrations):
            fig,axs=plt.subplots(1,3,figsize=(12,4))
            axs[0].imshow(rgb);axs[0].set_title(f"True {case['true_grade']} | Pred {case['pred_grade']}")
            axs[1].imshow(rgb);axs[1].imshow(cam,cmap="jet",alpha=.40,vmin=0,vmax=1)
            axs[1].set_title("Grad-CAM (predicted class)")
            axs[2].imshow(rgb)
            positive=np.maximum(occ,0)
            if positive.max()>0:positive=positive/(positive.max()+1e-8)
            axs[2].imshow(positive,cmap="magma",alpha=.40,vmin=0,vmax=1,
                          interpolation="nearest",extent=(0,size,size,0))
            axs[2].set_title("Occlusion sensitivity")
            for ax in axs:ax.axis("off")
            fig.suptitle(f"VALIDATION XAI | {filename} | NOT lesion labels",fontsize=10)
            fig.tight_layout()
            fig.savefig(folder/f"case_{k+1:02d}_{filename}.png",dpi=150)
            plt.close(fig)
    curves={name:[float(df[f"p_{name}_{fraction:.3f}"].mean()) for fraction in FRACTIONS]
            for name in METHODS}
    fig,ax=plt.subplots(figsize=(7,5))
    for name,prob in curves.items():ax.plot(np.array(FRACTIONS)*100,prob,marker="o",label=name)
    ax.set(xlabel="Image area replaced with blur (%)",
           ylabel="Mean original predicted-class probability",
           title="Validation deletion curves (lower = larger confidence drop)")
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out/"deletion_comparison.png",dpi=200);plt.close(fig)
    summary={
        "dataset_split":"internal MMRDR-CFP train-derived VALIDATION ONLY",
        "num_images":int(len(df)),
        "grade_counts":{str(k):int(v) for k,v in df.true_grade.value_counts().sort_index().items()},
        "non_prevalence_weighted_grade_balanced_sample":True,
        "image_size":size,"grid":args.grid,"fractions":list(FRACTIONS),
        "reference":"Gaussian blurred original RGB, kernel 41, sigma 12",
        "target":"original predicted class fixed across all perturbed images",
        "checkpoint_sha256":sha256(args.checkpoint),
        "best_epoch":ck.get("epoch"),
        "mean_integrated_confidence_drop":{k:float(df[f"aop_drop_{k}"].mean()) for k in METHODS},
        "paired_gradcam_minus_random":paired_bootstrap(df.aop_drop_gradcam,df.aop_drop_random,args.seed),
        "paired_occlusion_minus_gradcam":paired_bootstrap(df.aop_drop_occlusion,df.aop_drop_gradcam,args.seed),
        "median_gradcam_occlusion_spearman":float(df.gradcam_occlusion_spearman.median()),
        "median_gradcam_fc_randomized_spearman":float(sanity.fc_randomized_gradcam_spearman.median()),
        "sanity_images":int(len(sanity)),
        "limitations":[
            "Grade-balanced selected validation sample, not prevalence-weighted",
            "Occlusion ranks are favoured by the matched occlusion-based deletion metric",
            "Blurred occlusions create out-of-distribution images",
            "Classifier-only randomization is partial, not cascading or label-randomization",
            "No expert lesion masks or independent patient/site validation",
            "Attention at retinal edges/artifacts does not establish lesion localisation or safety",
        ]}
    write_json(out/"summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)
    print("Saved to",out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--val-csv",required=True)
    p.add_argument("--images-root",required=True)
    p.add_argument("--output-dir",default="outputs/xai_faithfulness")
    p.add_argument("--per-grade",type=int,default=10)
    p.add_argument("--grid",type=int,default=8)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--figures",type=int,default=6)
    p.add_argument("--sanity-images",type=int,default=15)
    run(p.parse_args())

if __name__=="__main__":main()
