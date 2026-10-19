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
    model=build_model(c["model"], c["pretrained"]).to(device)
    weights=None
    if c["weighted_loss"]:
        counts=np.bincount(train.grade, minlength=5)
        weights=torch.as_tensor(len(train)/(5*np.maximum(counts,1)),dtype=torch.float32,device=device)
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"])
    best=-1.0;stale=0;history=[]
    for epoch in range(1,c["epochs"]+1):
        model.train();sum_loss=0;total=0
        for xb,yb,_ in tqdm(train_dl,desc=f"epoch {epoch}",leave=False):
            xb=xb.to(device);yb=yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(xb),yb);loss.backward();optimizer.step()
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
            if stale>=c["patience"]:break
    pd.DataFrame(history).to_csv(out/"history.csv",index=False)
    write_json(out/"run_metadata.json", {"config":c,"python":platform.python_version(),"torch":torch.__version__,"device":str(device),"best_validation_macro_f1":best,"test_evaluated":False})
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
