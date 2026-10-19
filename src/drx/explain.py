"""ResNet Grad-CAM implementation for predicted class; no lesion localisation claims."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .core import build_model, device_choice, load_checkpoint
from .data import transform


def gradcam(model, x, target_class=None):
    """Gradient-weighted activation map from ResNet layer4."""
    activations, gradients = [], []
    def forward_hook(module, inp, output):
        activations.append(output.detach())
        output.register_hook(lambda grad: gradients.append(grad.detach()))
    hook=model.layer4.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits=model(x.requires_grad_(True))
            idx=int(logits.argmax(dim=1).item()) if target_class is None else int(target_class)
            if not 0 <= idx < 5:raise ValueError("target class must be 0..4")
            logits[0,idx].backward()
        weights=gradients[0].mean(dim=(2,3),keepdim=True)
        raw=(weights*activations[0]).sum(dim=1,keepdim=True)
        raw=F.relu(raw)
        raw=F.interpolate(raw,size=x.shape[-2:],mode="bilinear",align_corners=False)
        arr=raw[0,0].cpu().numpy()
        arr=(arr-arr.min())/(arr.max()-arr.min()+1e-8)
        return arr, idx, logits.detach().softmax(dim=1)[0].cpu().numpy()
    finally:
        hook.remove()


def run(args):
    device=device_choice();ck=load_checkpoint(args.checkpoint,device)
    model=build_model(ck["model_name"],False).to(device)
    model.load_state_dict(ck["model_state"]);model.eval()
    with Image.open(args.image) as im:
        im=im.convert("RGB")
        x=transform(ck["image_size"])(im).unsqueeze(0).to(device)
        array=np.array(im.resize((ck["image_size"],ck["image_size"])))
    cam,grade,prob=gradcam(model,x,args.target_class)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(array);axs[0].set_title("Input CFP")
    axs[1].imshow(cam,cmap="jet",vmin=0,vmax=1);axs[1].set_title(f"Grad-CAM for grade {grade}")
    axs[2].imshow(array);axs[2].imshow(cam,cmap="jet",alpha=.35,vmin=0,vmax=1)
    axs[2].set_title(f"Attribution overlay | P(grade {grade})={prob[grade]:.3f}")
    for a in axs:a.axis("off")
    fig.tight_layout();fig.savefig(out,dpi=150);plt.close(fig)
    print(f"Saved {out}. Saliency indicates model influence, not validated pathology.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--image",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--target-class",type=int)
    run(parser.parse_args())
if __name__=="__main__":main()
