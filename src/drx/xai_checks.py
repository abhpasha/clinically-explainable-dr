"""Validation-only XAI faithfulness utilities; never infer lesion localisation.

Use with `drx.explain.gradcam` for a fixed original predicted target class.
All deletion/occlusion comparisons must use the same image blur reference.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

FRACTIONS = (0.0, 0.125, 0.25, 0.375, 0.5)


def rank_regions(scores):
    values = np.asarray(scores, dtype=float).ravel()
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Importance scores must be finite.")
    return np.argsort(-values, kind="stable")


def replace_regions(image, baseline, ordered_regions, grid=8):
    """Blur the indexed disjoint image blocks, retaining all other pixels."""
    if image.ndim != 3 or image.shape != baseline.shape:
        raise ValueError("Expected same-shaped CHW input and baseline")
    side = image.shape[-1]
    if image.shape[-2] != side or side % grid:
        raise ValueError("Square image side must divide the patch grid")
    result = image.clone()
    for index in ordered_regions:
        row, col = divmod(int(index), grid)
        if not (0 <= row < grid and 0 <= col < grid):
            raise ValueError("Region index out of bounds")
        y0, y1 = row*side//grid, (row+1)*side//grid
        x0, x1 = col*side//grid, (col+1)*side//grid
        result[:, y0:y1, x0:x1] = baseline[:, y0:y1, x0:x1]
    return result


def target_probability(model, images, target, device, batch_size=8):
    """Probability of one FIXED original class across perturbed images."""
    values = []
    with torch.inference_mode():
        for batch in images.split(batch_size, dim=0):
            values.extend(model(batch.to(device)).softmax(1)[:, target].cpu().tolist())
    return np.asarray(values, dtype=float)


def occlusion_importance(model, image, baseline, target, original_p,
                         grid, device, batch_size=8):
    """Leave-one-block-out confidence change; grid^2 model perturbations."""
    values = []
    for start in range(0, grid*grid, batch_size):
        perturbed = torch.stack([
            replace_regions(image, baseline, [j], grid)
            for j in range(start, min(start+batch_size, grid*grid))
        ])
        values.extend(target_probability(model, perturbed, target, device, batch_size))
    return original_p - np.asarray(values)


def deletion_probabilities(model, image, baseline, rank, target, grid,
                           device, batch_size=8, fractions=FRACTIONS):
    """Predicted-class probability as top-ranked regions are blurred."""
    regions = grid*grid
    images = torch.stack([
        replace_regions(image, baseline, rank[:round(f*regions)], grid)
        for f in fractions
    ])
    return target_probability(model, images, target, device, batch_size)


def mean_integrated_drop(probabilities, fractions=FRACTIONS):
    """Signed area-averaged confidence drop; larger means stronger deletion."""
    p = np.asarray(probabilities, dtype=float)
    f = np.asarray(fractions, dtype=float)
    if len(p) != len(f) or len(p) < 2 or f[-1] <= f[0]:
        raise ValueError("Invalid curve")
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(p[0] - p, f)
    else:
        area = np.trapz(p[0] - p, f)
    return float(area / (f[-1] - f[0]))


def spearman_map_correlation(a, b):
    """Return NaN when one heatmap is constant."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Incompatible heatmaps")
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def paired_bootstrap(a, b, seed=42, repetitions=2000):
    """Paired image-resampling CI. Descriptive, not clinical validation."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if not len(diff):
        raise ValueError("No finite differences")
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(diff), size=(repetitions, len(diff)))
    ci = np.quantile(diff[selections].mean(1), [.025, .975])
    return {"n": int(len(diff)), "mean_difference": float(diff.mean()),
            "bootstrap_95_percentile_CI": [float(x) for x in ci]}
