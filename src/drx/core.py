import json
import random
from pathlib import Path

import numpy as np
import torch
from torchvision.models import (resnet18, resnet50, ResNet18_Weights, ResNet50_Weights)
import yaml

N_CLASSES = 5
GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]


def read_config(path):
    with open(path, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    if c["model"] not in ("resnet18", "resnet50"):
        raise ValueError("model must be resnet18 or resnet50")
    if not 0 < c["val_fraction"] < 1:
        raise ValueError("val_fraction must be between zero and one")
    return c


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(name="resnet50", pretrained=True):
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    elif name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(name)
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
    return model


def device_choice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
