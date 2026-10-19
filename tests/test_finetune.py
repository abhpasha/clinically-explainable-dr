import pytest
from drx.core import build_model
from drx.finetune import set_trainable, frozen_bn_eval

def test_modes():
    model = build_model("resnet18", False)
    head = set_trainable(model,"head")
    assert head == sum(p.numel() for p in model.fc.parameters())
    assert not model.layer4[0].conv1.weight.requires_grad
    partial = set_trainable(model,"partial")
    assert partial>head and model.layer4[0].conv1.weight.requires_grad
    assert not model.layer3[0].conv1.weight.requires_grad
    model.train()
    frozen_bn_eval(model)
    assert not model.bn1.training and not model.layer4[0].bn1.training
    assert set_trainable(model,"full")>partial
    with pytest.raises(ValueError):
        set_trainable(model,"invalid")
