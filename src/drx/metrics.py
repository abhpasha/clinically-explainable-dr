from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)

from .core import GRADE_NAMES, N_CLASSES, write_json


def expected_calibration_error(y_true, probabilities, bins=10):
    y_true = np.asarray(y_true)
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != N_CLASSES or len(p) != len(y_true) or len(p) == 0:
        raise ValueError("Expected non-empty N x 5 probability array")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Invalid probabilities")
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf < edges[i+1] if i < bins-1 else conf <= edges[i+1])
        if mask.any():
            ac, avg = float(correct[mask].mean()), float(conf[mask].mean())
            ece += mask.mean() * abs(ac - avg)
            rows.append((avg, ac, int(mask.sum())))
    return float(ece), rows


def summarise(y_true, probabilities, output_dir, paths=None):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or (y < 0).any() or (y >= N_CLASSES).any():
        raise ValueError("Invalid predictions or grades")
    pred = p.argmax(axis=1)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    report = classification_report(y, pred, labels=list(range(N_CLASSES)), output_dict=True, zero_division=0)
    ece, bins = expected_calibration_error(y, p)
    onehot = np.eye(N_CLASSES)[y]
    results = {
        "n": len(y), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(N_CLASSES)), average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred, labels=list(range(N_CLASSES)), weights="quadratic")),
        "expected_calibration_error_top_label_10_bins": ece,
        "multiclass_brier_score_sum_over_classes": float(np.mean(np.sum((p-onehot)**2,axis=1))),
        "undergraded": int((pred < y).sum()),
        "undergraded_by_at_least_two": int((pred <= y-2).sum()),
        "per_class": {str(i): report[str(i)] for i in range(N_CLASSES)},
        "confusion_matrix": cm.tolist(),
    }
    write_json(out / "metrics.json", results)
    pd.DataFrame(cm, index=GRADE_NAMES, columns=GRADE_NAMES).to_csv(out / "confusion_matrix.csv")
    df = pd.DataFrame({"path": paths if paths is not None else [f"sample_{i}" for i in range(len(y))],
                       "true_grade": y, "pred_grade": pred, "confidence": p.max(axis=1),
                       "undergraded_by": y-pred, "correct": y==pred})
    for i in range(N_CLASSES):
        df[f"prob_{i}"] = p[:,i]
    df.to_csv(out / "predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES), [str(i) for i in range(N_CLASSES)])
    ax.set_yticks(range(N_CLASSES), [str(i) for i in range(N_CLASSES)])
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out/"confusion_matrix.png", dpi=180);plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,5))
    if bins:
        ax.scatter([a for a,_,_ in bins],[b for _,b,_ in bins],s=[max(25,c*2) for *_,c in bins])
    ax.plot([0,1],[0,1],"k--", alpha=.6)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean confidence",ylabel="Empirical accuracy",title="Top-label reliability (10 bins)")
    fig.tight_layout();fig.savefig(out/"reliability.png",dpi=180);plt.close(fig)
    return results
