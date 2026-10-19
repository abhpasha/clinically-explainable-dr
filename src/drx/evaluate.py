import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .core import build_model, device_choice, load_checkpoint
from .data import CFPDataset, parse_manifest
from .metrics import summarise
from .train import collect


def run(args):
    device=device_choice()
    ckpt=load_checkpoint(args.checkpoint,device)
    model=build_model(ckpt["model_name"],pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    manifest=parse_manifest(args.test_csv,args.test_images,args.filename_col,args.label_col)
    dl=DataLoader(CFPDataset(manifest,ckpt["image_size"]),batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    y,p,paths=collect(model,dl,device)
    result=summarise(y,p,args.output_dir,[Path(q).name for q in paths])
    print("OFFICIAL HELD-OUT TEST:",result)


def main():
    parser=argparse.ArgumentParser(description="Run only after model selection is frozen")
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--test-csv",required=True)
    parser.add_argument("--test-images",required=True)
    parser.add_argument("--filename-col")
    parser.add_argument("--label-col",default="grade")
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--num-workers",type=int,default=2)
    parser.add_argument("--output-dir",default="outputs/test")
    run(parser.parse_args())
if __name__=="__main__":main()
