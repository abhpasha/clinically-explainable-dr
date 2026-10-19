# Clinically Explainable AI for Diabetic Retinopathy Grading

**Ahsan Bin Hasan Pasha** · PyTorch · ResNet-50 · Grad-CAM · Reliability and clinical error analysis

> **Research prototype only; not a medical device or diagnostic system.** The source code and synthetic smoke tests are implemented. Genuine MMRDR training runs, calibrated interpretations and held-out test results are **not yet completed**. No performance or clinical-safety claims should be made before running the dataset.

## Aim

Predict diabetic retinopathy (DR) severity from MMRDR colour fundus photographs (CFP) with five grades: **0** no DR, **1** mild non-proliferative DR (NPDR), **2** moderate NPDR, **3** severe NPDR and **4** proliferative DR (PDR). Evaluate per-class performance, clinically significant **under-grading**, model confidence and Grad-CAM visual explanations.

## Dataset access and critical caveat

- Dataset paper: https://www.nature.com/articles/s41597-026-07005-9
- Download: https://figshare.com/articles/dataset/MMRDR/29423747
- Authors' evaluation code: https://github.com/Vladimirovich2019/MMRDR_Evaluation

Obtain the **MMRDR-CFP** subset and inspect its actual directory layout, `FP.csv` filenames and grade column. Official CFP train/test partitions are described as **image-level**, not confirmed patient-independent: filename and path checks cannot establish patient independence. Keep all test labels unavailable during model choice and only evaluate the official test set after freezing the trained model.

Never commit MMRDR images, personally identifiable data, weights or assessment-restricted materials. `.gitignore` excludes local datasets and generated artifacts.

## Installation

Python 3.10+; a GPU is recommended for the ResNet-50 training configuration.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
~~~

For GPU installs, select compatible torch/torchvision packages from https://pytorch.org/get-started/locally/. Pretrained ImageNet weights download on the first run; offline operation requires cached weights.

## Expected local dataset layout

~~~text
data/MMRDR-CFP/
  train/FP.csv
  train/img/tr000001.jpg
  test/FP.csv
  test/img/ts000001.jpg
~~~

Replace example folders and filename column with **actual** dataset values. If `FP.csv` contains `filename` rather than `image`, change `--filename-col`. Code refuses ambiguous CSV columns and missing/unresolvable images rather than guessing.

## 1. Inspect counts and overlap

~~~bash
python -m drx.inspect \
  --train-csv data/MMRDR-CFP/train/FP.csv --train-images data/MMRDR-CFP/train/img \
  --test-csv data/MMRDR-CFP/test/FP.csv --test-images data/MMRDR-CFP/test/img \
  --filename-col image --label-col grade
~~~

This checks counts and exact path/filename collisions; it does **not** check patient identities or near-duplicate photos.

## 2. Train: official training set only

~~~bash
python -m drx.train \
  --config configs/resnet50.yaml \
  --train-csv data/MMRDR-CFP/train/FP.csv --train-images data/MMRDR-CFP/train/img \
  --test-images data/MMRDR-CFP/test/img \
  --filename-col image --label-col grade
~~~

The seeded, stratified training/validation split is made **inside the official training set**. The model is selected by validation macro-F1 with early stopping. Configured ResNet-50 uses ImageNet initialization, 384px inputs, training-only augmentation, weighted cross entropy, AdamW and eight epochs. The official test CSV is not read during training. Best checkpoint: `outputs/resnet50/best.pt`.

For a CPU-only **software smoke test**, use `--config configs/quick_cpu.yaml`; this is not a substitute for actual full-scale evaluation.

## 3. Held-out test evaluation (once model selection is frozen)

~~~bash
python -m drx.evaluate \
  --checkpoint outputs/resnet50/best.pt \
  --test-csv data/MMRDR-CFP/test/FP.csv --test-images data/MMRDR-CFP/test/img \
  --filename-col image --label-col grade \
  --output-dir outputs/test
~~~

Outputs: `metrics.json`, `predictions.csv`, confusion matrix CSV and PNG, reliability plot. Report: accuracy, macro-F1, grade 0–4 precision/recall/F1, quadratic-weighted Cohen kappa, grade underestimation (including differences of 2+), top-label expected calibration error and multiclass Brier score. Calibration scores are descriptive and do not demonstrate safety.

## 4. Explain genuine examples using Grad-CAM

~~~bash
python -m drx.explain \
  --checkpoint outputs/resnet50/best.pt \
  --image data/MMRDR-CFP/test/img/REAL_IMAGE.jpg \
  --output outputs/test/example_gradcam.png
~~~

Generate 4–8 examples from real predictions, including correct grades, confident errors and severe under-grades if present. The overlay shows class-relevant activation, **not validated lesion segmentation or a clinician's reasoning**. The seven provided lesion indicators, where present, are image-level annotations, not pixel masks.

## Code layout

| Path | Purpose |
| --- | --- |
| `src/drx/data.py` | Manifest parsing, five-class grade checks, augmentations, overlap detection |
| `src/drx/core.py` | Model, seeding and configuration |
| `src/drx/inspect.py` | File-level data audit |
| `src/drx/train.py` | Training and training-only validation |
| `src/drx/evaluate.py` | Held-out test predictions |
| `src/drx/metrics.py` | Performance, confusion matrix, error severity, reliability |
| `src/drx/explain.py` | ResNet Grad-CAM overlays |
| `configs/` | Full model and lightweight CPU configurations |
| `tests/` | Synthetic unit and integration smoke tests |
| `report/REPORT_TEMPLATE.md` | 2,000-word assessment report outline |

## Reproducibility and limits

Seed 42, configuration-controlled training, saved train/validation manifests, checkpoint and run metadata, explicit test-only evaluation. Reproducing exact numbers requires the actual MMRDR data, model weights, dependencies and compute. Seeded runs can still differ across devices/libraries.

Published MMRDR CFP patient identifiers may not be available: no claim of patient-independent evaluation. No external hospital validation, clinical review, image-quality or deployment evaluation has been performed. Models may emphasise imaging artifacts. Interpret any high confidence and Grad-CAM heatmap cautiously.

## Citations

MMRDR: https://doi.org/10.1038/s41597-026-07005-9 · DR grades: https://pubmed.ncbi.nlm.nih.gov/13129861/ · ResNet: https://arxiv.org/abs/1512.03385 · Grad-CAM: https://doi.org/10.1109/ICCV.2017.74 · Saliency checks: https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html

Original research pipeline for an assessment, developed with AI assistance. Check your institution's requirements for acknowledging such assistance.

**Submission status: pipeline and tests available; actual data audit, model training, performance findings, Grad-CAM examples and final report pending.**
