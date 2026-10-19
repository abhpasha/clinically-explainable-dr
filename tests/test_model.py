import torch
from drx.core import build_model
from drx.explain import gradcam


def test_backbone_shape_and_cam():
    torch.set_num_threads(2)
    model=build_model('resnet18',False).eval()
    x=torch.randn(1,3,64,64)
    with torch.no_grad():
        assert tuple(model(x).shape)==(1,5)
    cam,grade,prob=gradcam(model,x)
    assert cam.shape==(64,64)
    assert 0 <= grade <= 4
    assert abs(prob.sum()-1)<1e-5
