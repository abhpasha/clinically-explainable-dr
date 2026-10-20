# Clinically Explainable AI for Diabetic Retinopathy Grading

**Ahsan Bin Hasan Pasha** · PyTorch · ResNet-50 · Grad-CAM · Reliability and clinical error analysis

> **Research prototype only; not a medical device or diagnostic system.** The 224-pixel MMRDR-CFP Kaggle experiment has now been completed. These results are a single image-level-split research experiment, not external or patient-independent clinical validation.

## Aim

Predict diabetic retinopathy (DR) severity from MMRDR colour fundus photographs (CFP) with five grades: **0** no DR, **1** mild non-proliferative DR (NPDR), **2** moderate NPDR, **3** severe NPDR and **4** proliferative DR (PDR). Evaluate per-class performance, clinically significant **under-grading**, model confidence and Grad-CAM visual explanations.

## Dataset access and critical caveat

- Dataset paper: https://www.nature.com/articles/s41597-026-07005-9
- Download: https://figshare.com/articles/dataset/MMRDR/29423747
- Authors' evaluation code: https://github.com/Vladimirovich2019/MMRDR_Evaluation

Obtain the **MMRDR-CFP** subset and inspect its actual directory layout, `FP.csv` filenames and grade column. Official CFP train/test partitions are described as **image-level**, not confirmed patient-independent: filename and path checks cannot establish patient independence. Keep all test labels unavailable during model choice and only evaluate the official test set after freezing the trained model.

Never commit MMRDR images, personally identifiable data, weights or assessment-restricted materials. `.gitignore` excludes local datasets and generated artifacts.

## New: 19 GB dataset preparation + staged transfer learning

**[Read the step-by-step Colab guide](docs/MMRDR_CFP_19GB_COLAB.md)** before downloading. The official MMRDR data are shipped in nine split archive volumes (approximately 18.6 GB total). All parts are required; extract **CFP only** with sufficient free Colab disk. The original CFP dataset may contain **one** `MMRDR-CFP/FP.csv` containing both `tr######.jpg` and `ts######.jpg` records—not necessarily separate train/test CSVs.

If your extracted dataset has one `FP.csv` and one `img/` folder, run:

```bash
python -m drx.prepare --csv /content/mmrdr_data/MMRDR-CFP/FP.csv --images /content/mmrdr_data/MMRDR-CFP --output data/manifests
python -m drx.train --config configs/resnet50_head.yaml --train-csv data/manifests/train.csv --train-images /content/mmrdr_data/MMRDR-CFP
python -m drx.train --config configs/resnet50_partial.yaml --train-csv data/manifests/train.csv --train-images /content/mmrdr_data/MMRDR-CFP
```

Head-only is a frozen ImageNet-pretrained backbone with a trained five-grade classifier. Partial fine-tuning then loads the head-only checkpoint and trains the final residual block plus classifier. **Both use the same seeded validation partition**, with the official test held out. Optional higher-resolution stage: `configs/resnet50_partial_384.yaml`. The original `configs/resnet50.yaml` remains a full fine-tuning alternative. The JPEGs are loaded, resized, normalised and augmented **on the fly**; no 19 GB preprocessing copy is created.

> The commands below under "Expected local dataset layout" show an **alternative split-folder arrangement**. Prefer the new Colab guide and inspect the actual extracted paths. The 224-pixel experiment is complete; the 384-pixel extension is a separate validation-only experiment.

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

**Submission status (20 October 2026):** 224-pixel data audit, frozen-head baseline (4 epochs), partial fine-tuning (6 epochs), final held-out test and illustrative Grad-CAM completed. Report remains to be finalised. Higher-resolution follow-up is planned separately.


## Completed 224-pixel CFP experiment (20 October 2026)

- Dataset: 11,118 CFP images; official training 8,893; official held-out test 2,225.
- Stratified internal validation: 1,334 images from the official training portion, random seed 42.
- Frozen ImageNet ResNet-50 head: four epochs; validation accuracy **0.6432**, macro-F1 **0.5211**, quadratic weighted kappa **0.7785**.
- Partial layer4 + head: six epochs; best validation checkpoint at epoch four; validation accuracy **0.7571**, macro-F1 **0.6293**, QWK **0.8483**.
- **One-time test evaluation of the validation-selected partial model:** accuracy **0.7299**, macro-F1 **0.5979**, QWK **0.8354** (n=2,225).
- Test errors: **366** undergraded cases, including **97** undergraded by 2+ ordinal grades. Moderate NPDR recall **0.3594**, PDR recall **0.5821**.
- The five-grade CFP labels and seven image-level lesion indicators are not pixel segmentation ground truth; Grad-CAM is model attribution, not verified lesion localisation.
- Important: this completed experiment ran in a self-contained Kaggle notebook aligned with, but not identical to, the repository CLI. Any results reported from future repository-CLI experiments must be distinguished and include their own configs, histories and seed/version metadata.
- Higher-resolution follow-up configurations: `configs/resnet50_head_224_long.yaml`, `configs/resnet50_head_384_extended.yaml` and `configs/resnet50_partial_384_extended.yaml`.
- **No repeated testing on the official test set for tuning:** compare follow-up models on the same internal validation split; record selection before any new final held-out test evaluation. The previously inspected test performance is observational, not a fresh blind assessment.

The downloadable Kaggle experiment output archive contains checkpoint weights and prediction-level data; **do not commit it to GitHub**. Publish only source, configuration, metadata and selected aggregate results/figures consistent with redistribution permissions.

## Kaggle T4 reproducibility, mixed precision and recovery

The October 2026 Kaggle extension clones a **pinned Git commit** and installs this repository in editable mode without replacing Kaggle's CUDA-enabled PyTorch. It checks runtime packages against the repository's `requirements.txt` and records package versions and configuration hashes, so the environment can be inspected rather than assumed reproducible.

The extended configs have `use_amp: true`, enabling automatic mixed precision on a CUDA GPU (one T4 is used even when Kaggle offers T4 x2), and `resume: true`, saving **`last.pt` with optimizer, scaler, training history, epoch and PyTorch RNG state after each completed epoch**. The best validation checkpoint remains separate in `best.pt`. Per-epoch history and `progress.json` are updated continuously; `run_metadata.json` indicates a completed stage. Resume is supported only when the actual checkpoint files have survived or have been restored to the configured output directory. Terminating a Kaggle session can remove ephemeral `/kaggle/working` state—download intermediate archives or save notebook output versions to preserve files.

Default extension sequence: 224px frozen head up to **12 epochs** (patience 4); 384px frozen head up to **8 epochs** (patience 4); 384px partial layer4+classifier up to **8 epochs** (patience 3). Those are maxima; early stopping selects the best validation epoch. The higher resolution uses the same official training subset and internal validation partition as the original experiment; it does not read official test labels for model choice. Comparing validation scores across exploratory stages is not a new blinded test. GPU availability, quota, run time and final predictive performance cannot be guaranteed.

The notebook exports model checkpoints and aggregate results in ZIP archives **without original retinal images**. Lesion indicators are image-level and Grad-CAM is illustrative attribution, not spatial pathology validation. Do not publish trained weights or individual sample outputs if source-data permissions prohibit redistribution.

## Completed exploratory extension: longer head and 384px (20 October 2026)

A second Kaggle notebook (Git-pinned at `9d7aa84c7a2f1b3a632c53aae232f44a08e072be`) completed all three stages using one NVIDIA T4 with automatic mixed precision, the same 8,893-image official training portion and the same seeded 1,334-image internal validation split. These are **validation results only**:

| Stage | Epoch selected | Validation accuracy | Validation macro-F1 | Validation QWK | Top-label ECE | Validation images undergraded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 224px frozen ResNet50, extended | 7 | 0.684408 | 0.581130 | 0.801429 | 0.082425 | 181/1334 |
| 384px frozen ResNet50 | 7 | 0.694153 | 0.580442 | 0.807639 | 0.084713 | 185/1334 |
| 384px ResNet50 layer4 + head | 6 | 0.752624 | **0.676908** | **0.853092** | 0.028804 | **110/1334** |

The 384px partially fine-tuned model is the validation-selected candidate. **The extended notebook did not evaluate it on the official held-out test set.** These exploratory validation results cannot establish clinical safety, patient-level generalisation or a true gain attributable purely to input resolution: compared experiments also differ in optimisation/training settings. Confidence calibration statistics are descriptive and obtained from a dataset used for model selection. If a final test evaluation is subsequently run, freeze the selected model and decision rules beforehand, report that the official test set was used previously for the historical 224px experiment, and do not tune again against test performance.

Recommended clinically oriented follow-up using saved predictions (no additional GPU training): per-grade recall, true-grade-stratified undergrading, risk-versus-coverage (confidence-based review), and a clearly labelled exploratory grade>=2 threshold analysis. Clearly distinguish attribution from lesion annotations and discuss clinically consequential severe-grade misses.

