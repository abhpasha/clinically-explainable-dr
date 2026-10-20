import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .core import build_model, device_choice, read_config, seed_all, write_json
from .data import CFPDataset, make_split, parse_manifest
from .finetune import set_trainable, frozen_bn_eval


def collect(model, loader, device):
    model.eval()
    ys, probs, paths = [], [], []
    with torch.inference_mode():
        for xb, yb, names in loader:
            logits = model(xb.to(device))
            ys.extend(yb.tolist()); probs.extend(logits.softmax(dim=1).cpu().numpy()); paths.extend(names)
    return np.array(ys), np.array(probs), paths


def run(args):
    c = read_config(args.config); seed_all(c["seed"])
    out = Path(c["output_dir"]);out.mkdir(parents=True,exist_ok=True)
    tr = parse_manifest(args.train_csv, args.train_images, args.filename_col, args.label_col)
    if args.test_images:
        test_images = {f.name for f in Path(args.test_images).rglob("*") if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")}
        overlap = set(Path(f).name for f in tr.path) & test_images
        if overlap:
            raise ValueError(f"Train/test filename collision: {len(overlap)}")
    train, val = make_split(tr, c["val_fraction"], c["seed"])
    for name, part in (("train",train),("val",val)):
        pd.DataFrame({"filename":[Path(f).name for f in part.path],"grade":part.grade}).to_csv(out/f"{name}_split.csv",index=False)
    write_json(out/"dataset_summary.json", {"counts": {name:part.grade.value_counts().sort_index().to_dict() for name,part in (("train",train),("val",val))}, "notes":"CFP official test is image-level; patient independence cannot be guaranteed by published MMRDR split."})
    dl_args = {"batch_size": c["batch_size"], "num_workers": c["num_workers"]}
    train_dl=DataLoader(CFPDataset(train,c["image_size"], True),shuffle=True,**dl_args)
    val_dl=DataLoader(CFPDataset(val,c["image_size"]),shuffle=False,**dl_args)
    device=device_choice()
    from .core import load_checkpoint
    last_path=out/"last.pt"
    resume=bool(c.get("resume", False)) and last_path.is_file()
    # Resuming avoids a redundant ImageNet download and starts from the last completed epoch.
    model=build_model(c["model"], c["pretrained"] and not resume).to(device)
    if c.get("init_checkpoint") and not resume:
        previous = load_checkpoint(c["init_checkpoint"], device)
        if previous["model_name"] != c["model"]:
            raise ValueError("Checkpoint architecture does not match")
        model.load_state_dict(previous["model_state"])
    mode = c.get("train_mode", "full")
    trainable = set_trainable(model, mode)
    print(f"Training mode: {mode}; trainable parameters: {trainable:,}")
    weights=None
    if c["weighted_loss"]:
        counts=np.bincount(train.grade, minlength=5)
        weights=torch.as_tensor(len(train)/(5*np.maximum(counts,1)),dtype=torch.float32,device=device)
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=c["learning_rate"],weight_decay=c["weight_decay"])
    amp=bool(c.get("use_amp", False)) and device.type=="cuda"
    scaler=torch.cuda.amp.GradScaler(enabled=amp)
    best=-1.0;stale=0;history=[];start_epoch=1
    if resume:
        prior=load_checkpoint(last_path,device)
        if prior["model_name"]!=c["model"] or int(prior["image_size"])!=int(c["image_size"]):
            raise ValueError("Resume checkpoint architecture or resolution mismatch")
        model.load_state_dict(prior["model_state"])
        optimizer.load_state_dict(prior["optimizer_state"])
        if amp and prior.get("scaler_state"):
            scaler.load_state_dict(prior["scaler_state"])
        history=prior["history"];best=float(prior["best_validation_macro_f1"])
        stale=int(prior["stale"]);start_epoch=int(prior["epoch"])+1
        torch.set_rng_state(prior["torch_rng_state"].cpu())
        if device.type=="cuda" and prior.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in prior["cuda_rng_states"]])
        print(f"Resuming {out}: epoch {start_epoch}/{c['epochs']}, best validation macro-F1={best:.4f}",flush=True)
    print(f"AMP mixed precision enabled: {amp}", flush=True)
    for epoch in range(start_epoch,c["epochs"]+1):
        model.train();frozen_bn_eval(model);sum_loss=0;total=0
        for xb,yb,_ in tqdm(train_dl,desc=f"epoch {epoch}",leave=False):
            xb=xb.to(device);yb=yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss=criterion(model(xb),yb)
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            sum_loss+=loss.item()*len(yb);total+=len(yb)
        vy,vp,_=collect(model,val_dl,device)
        macro=float(f1_score(vy,vp.argmax(axis=1),average="macro",labels=list(range(5)),zero_division=0))
        record={"epoch":epoch,"train_loss":sum_loss/max(total,1),"val_macro_f1":macro}
        history.append(record);print(record,flush=True)
        if macro > best:
            best=macro;stale=0
            torch.save({"model_state":model.state_dict(),"model_name":c["model"],"image_size":c["image_size"],"epoch":epoch,"val_macro_f1":macro},out/"best.pt")
        else:
            stale+=1
        # Persist history and a full resumable optimizer snapshot after EVERY completed epoch.
        # The best.pt remains the validation-selected checkpoint, not the last checkpoint.
        pd.DataFrame(history).to_csv(out/"history.csv",index=False)
        torch.save({
            "model_state":model.state_dict(),"model_name":c["model"],
            "image_size":c["image_size"],"optimizer_state":optimizer.state_dict(),
            "scaler_state":scaler.state_dict() if amp else {},
            "epoch":epoch,"best_validation_macro_f1":best,"stale":stale,
            "history":history,"torch_rng_state":torch.get_rng_state(),
            "cuda_rng_states":torch.cuda.get_rng_state_all() if device.type=="cuda" else None,
        },last_path)
        write_json(out/"progress.json",{"last_completed_epoch":epoch,
            "best_validation_macro_f1":best,"stale":stale,
            "completed":False,"amp_enabled":amp})
        if stale>=c["patience"]:break
    write_json(out/"run_metadata.json", {"config":c,"python":platform.python_version(),"torch":torch.__version__,"device":str(device),"best_validation_macro_f1":best,"train_mode":mode,"trainable_parameters":trainable,"initial_checkpoint":c.get("init_checkpoint"),"amp_enabled":amp,"test_evaluated":False})
    write_json(out/"progress.json",{"last_completed_epoch":history[-1]["epoch"] if history else 0,
        "best_validation_macro_f1":best,"stale":stale,"completed":True,"amp_enabled":amp})
    print(f"Best validation macro-F1: {best:.4f}. Test deliberately NOT evaluated. Saved {out/'best.pt'}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/resnet50.yaml")
    parser.add_argument("--train-csv",required=True)
    parser.add_argument("--train-images",required=True)
    parser.add_argument("--test-images",default=None)
    parser.add_argument("--filename-col",default=None)
    parser.add_argument("--label-col",default="grade")
    run(parser.parse_args())

if __name__=="__main__":main()
