"""Transfer-learning modes: train only head, layer4+head, or full model."""
import torch

def set_trainable(model, mode):
    if mode not in ("head", "partial", "full"):
        raise ValueError("train_mode must be head, partial or full")
    for p in model.parameters():
        p.requires_grad = mode == "full"
    if mode in ("head", "partial"):
        for p in model.fc.parameters():
            p.requires_grad = True
    if mode == "partial":
        for p in model.layer4.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def frozen_bn_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
