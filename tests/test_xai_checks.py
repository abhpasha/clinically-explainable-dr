"""Synthetic XAI utility tests; do not require MMRDR images."""
import numpy as np
import pytest
import torch
from drx.xai_checks import (
    FRACTIONS, rank_regions, replace_regions, deletion_probabilities,
    mean_integrated_drop, paired_bootstrap, spearman_map_correlation,
)

class Tiny(torch.nn.Module):
    def forward(self, x):
        # Fixed features permit simple inference tests, not medical claims.
        signal=x.mean(dim=(1,2,3))
        return torch.stack([signal, -signal, torch.zeros_like(signal),signal*0,signal*0],dim=1)

def test_rank_and_mask_fixed_blocks():
    assert rank_regions([1,3,2]).tolist()==[1,2,0]
    x=torch.ones(3,8,8)
    bg=torch.zeros_like(x)
    altered=replace_regions(x,bg,[0,3],grid=2)
    assert altered[:,0:4,0:4].sum()==0
    assert altered[:,4:8,4:8].sum()==0
    assert altered[:,0:4,4:8].sum()==48

def test_deletion_curve_preserves_target_class():
    x=torch.ones(3,8,8)
    ref=torch.zeros_like(x)
    p=deletion_probabilities(Tiny(),x,ref,np.arange(4),0,2,torch.device('cpu'),batch_size=3)
    assert len(p)==len(FRACTIONS)
    assert p[0]>p[-1]
    assert mean_integrated_drop(p)>0

def test_rank_correlation_nan_for_flat():
    assert np.isnan(spearman_map_correlation([1,1],[0,1]))
    assert spearman_map_correlation([0,1,2],[0,1,2]) == pytest.approx(1)

def test_paired_bootstrap_is_deterministic():
    a=np.array([2.,3.,4.,5.])
    b=np.array([1.,2.,3.,4.])
    x=paired_bootstrap(a,b,seed=42)
    assert x['mean_difference']==pytest.approx(1)
    assert x['bootstrap_95_percentile_CI']==pytest.approx([1,1])
