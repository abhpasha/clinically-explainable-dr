# Clinically Explainable AI for Diabetic Retinopathy Grading

> **Submission map:** Primary reported results come from experiment commit `0713e20bb81741163ffea2ee39a5be3dbb870b86`. Later commits add the cleaned notebook, evidence CSV/JSON exports, reproducibility scripts and documentation; they do not change the reported seed-42 model or official-test predictions.

**Ahsan Bin Hasan Pasha** · PyTorch · ResNet-50 · Grad-CAM · reliability and clinical error analysis

> **Research prototype only. Not a medical device or autonomous diagnostic system.**

This repository contains the code and configuration used for the final Aston University × Boots Opticians KTP practical assessment on five-class diabetic retinopathy (DR) grading from MMRDR colour fundus photographs (CFP).

## Final experiment

The final protocol uses the MMRDR CFP subset (11,118 images: 8,893 official training and 2,225 official test images).

Before model fitting, decoded RGB pixel hashes are used to:
- remove 62 same-label exact duplicate training images;
- exclude 34 official-training images that exactly match official-test pixels;
- verify no exact-pixel overlap between the final development, validation and official-test partitions.

The resulting clean training pool contains **8,797** images and is split once with seed 42 into:
- development: **7,477**
- validation: **1,320**
- official test: **2,225** (kept outside model selection)

CFP does not provide patient identifiers, so patient-level independence cannot be established.

## Model comparison

Four ImageNet-pretrained ResNet-50 transfer-learning configurations are evaluated on the same clean split:

| Model | Resolution | Training |
| --- | ---: | --- |
| A | 224 px | classifier head only |
| B | 224 px | layer4 + classifier |
| C | 384 px | classifier head only |
| D | 384 px | layer4 + classifier |

Validation macro-F1 is the primary selection metric; QWK and fewer undergrades are predefined tie-breakers. The selected primary model is **D: 384 px partial fine-tuning**.

Final seed-42 validation: accuracy **0.755**, macro-F1 **0.626**, QWK **0.851**, ECE **0.035**.

Qualified follow-up official test: accuracy **0.746**, macro-F1 **0.602**, QWK **0.830**. This test is explicitly **non-blind** because an earlier exploratory model had previously been evaluated on the same official test partition. No model, threshold or XAI setting was changed after final validation selection.

## Installation

Python 3.10+ and a CUDA GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
```

## Dataset

Dataset paper: https://doi.org/10.1038/s41597-026-07005-9

Obtain MMRDR-CFP separately. Do **not** commit the dataset to this repository.

Expected extracted layout:

```text
MMRDR-CFP/
  FP.csv
  img/
    tr000001.jpg
    ...
    ts000001.jpg
```

## Reproduce the final clean split

```bash
python -m drx.clean_split \
  --csv /path/to/MMRDR-CFP/FP.csv \
  --images /path/to/MMRDR-CFP \
  --output data/final_clean \
  --seed 42 \
  --val-fraction 0.15
```

This writes:
- `development.csv`
- `validation.csv`
- `official_test.csv`
- exact-pixel hash audit tables
- `clean_split_summary.json`

## Train the four final configurations

All stages use the **same fixed manifests**.

```bash
python -m drx.train --config configs/final_head_224_clean.yaml \
  --train-csv data/final_clean/development.csv \
  --val-csv data/final_clean/validation.csv \
  --train-images /path/to/MMRDR-CFP --filename-col image --label-col grade

python -m drx.train --config configs/final_partial_224_clean.yaml \
  --train-csv data/final_clean/development.csv \
  --val-csv data/final_clean/validation.csv \
  --train-images /path/to/MMRDR-CFP --filename-col image --label-col grade

python -m drx.train --config configs/final_head_384_clean.yaml \
  --train-csv data/final_clean/development.csv \
  --val-csv data/final_clean/validation.csv \
  --train-images /path/to/MMRDR-CFP --filename-col image --label-col grade

python -m drx.train --config configs/final_partial_384_clean.yaml \
  --train-csv data/final_clean/development.csv \
  --val-csv data/final_clean/validation.csv \
  --train-images /path/to/MMRDR-CFP --filename-col image --label-col grade
```

The 384 px stages use physical batch size 8 with gradient accumulation 2; effective batch size is 16 across all four configurations. BatchNorm layers remain in evaluation mode during transfer-learning stages. Partial models receive the frozen-head stage plus the partial stage, so comparisons also reflect additional optimisation.

## Evaluate a frozen checkpoint

Only evaluate the official test after model selection is frozen.

```bash
python -m drx.evaluate \
  --checkpoint outputs/final_partial_384_clean/best.pt \
  --test-csv data/final_clean/official_test.csv \
  --test-images /path/to/MMRDR-CFP \
  --filename-col image --label-col grade \
  --output-dir outputs/final_test
```

The final assessment additionally froze uncertainty-review thresholds on validation before applying them unchanged to the official test.

## Explainability

Grad-CAM is implemented locally in `src/drx/explain.py` using ResNet layer4 activations and gradients. `src/drx/xai_checks.py` implements the reusable deletion, occlusion, correlation and bootstrap utilities. The final fundus-mask restriction, repeated random rankings and cascading parameter-randomisation orchestration were run in the final Kaggle analysis notebook and are not hidden inside `xai_checks.py`.

Example:

```bash
python -m drx.explain \
  --checkpoint outputs/final_partial_384_clean/best.pt \
  --image /path/to/MMRDR-CFP/img/REAL_IMAGE.jpg \
  --output outputs/example_gradcam.png
```

Grad-CAM is model attribution, **not lesion segmentation or proof of clinical correctness**.

## Final result summary

| Split | Accuracy | Macro-F1 | QWK |
| --- | ---: | ---: | ---: |
| Clean validation | 0.755 | 0.626 | 0.851 |
| Follow-up official test | 0.746 | 0.602 | 0.830 |

Test-set clinical error highlights:
- 306 undergrades;
- 91 undergraded by two or more grades;
- 37/134 PDR cases undergraded;
- exploratory grade ≥2 sensitivity 0.760 (95% CI 0.726–0.792);
- 171/902 DR images predicted as No DR;
- 102/1,323 No-DR images upgraded.

At approximately 30–33% review burden, validation-frozen uncertainty policies captured about 59–62% of PDR undergrades on the official test, leaving 14–15 of 37 PDR undergrades outside review.

## Resources and reproducibility

- Model: torchvision ResNet-50 with `IMAGENET1K_V2` pretrained weights.
- Framework: PyTorch / torchvision.
- Metrics and statistical analysis: scikit-learn, SciPy, NumPy and pandas.
- Grad-CAM: repository implementation in `src/drx/explain.py`, based on the method of Selvaraju et al. (2017).
- XAI sanity checks: repository implementation informed by Adebayo et al. (2018).
- Configuration files: `configs/final_*_clean.yaml`.
- Seed: 42 for the primary protocol; seeds 7 and 123 are post-hoc sensitivity runs only.

Portable dependency ranges are in `requirements.txt`; exact core package versions recorded from the final Kaggle run are in `requirements-lock.txt`. GPU runs can still vary slightly across hardware and library builds.

## Code layout

| Path | Purpose |
| --- | --- |
| `src/drx/clean_split.py` | decoded-pixel duplicate handling and fixed split creation |
| `src/drx/data.py` | manifest parsing and transforms |
| `src/drx/core.py` | model construction, seeding, configuration |
| `src/drx/train.py` | fixed-manifest training, AMP, accumulation, resume |
| `src/drx/evaluate.py` | frozen-checkpoint evaluation |
| `src/drx/metrics.py` | performance, undergrading and calibration |
| `src/drx/explain.py` | custom ResNet Grad-CAM |
| `src/drx/xai_checks.py` | quantitative XAI faithfulness helpers |
| `configs/final_*_clean.yaml` | final four model configurations |
| `requirements-lock.txt` | exact core package versions recorded from the final Kaggle run |
| `notebooks/final_clean_experiment_submission.ipynb` | cleaned, output-stripped end-to-end Kaggle orchestration for the final experiment and report analyses |
| `scripts/final_report_analysis.py` | reproduce report-level CIs, clinical errors, calibration and Mild NPDR pattern analysis from generated predictions |
| `tests/` | synthetic unit and smoke tests |

## Limitations

The final primary comparison uses one training seed; validation is used for both epoch and model selection; hyperparameters were informed by earlier exploratory work on the same official training pool; the official test is not historically blind; patient-level independence is unavailable; and the explanation methods do not establish lesion localisation or clinical safety.

## References

- Tang et al. (2026), MMRDR: https://doi.org/10.1038/s41597-026-07005-9
- He et al. (2016), ResNet: https://doi.org/10.1109/CVPR.2016.90
- Selvaraju et al. (2017), Grad-CAM: https://doi.org/10.1109/ICCV.2017.74
- Adebayo et al. (2018), saliency sanity checks: https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html


## Final evidence exports

Small, non-restricted aggregate outputs used by the report are committed under `results/final/`, including the controlled model comparison, paired-bootstrap comparisons, review-policy results, lesion-conditioned undergrading, label/lesion consistency audit, XAI summary and parameter-randomisation sanity checks, within-PDR lesion probe, extra-seed sensitivity and test clinical-error summary. These are post-run evidence exports; they do not alter model selection or permit test-set tuning.

The cleaned notebook is committed with outputs stripped. The **primary experimental code remains pinned to commit `0713e20bb81741163ffea2ee39a5be3dbb870b86`**; subsequent commits add documentation, reproducibility wrappers and small exported results without changing the reported seed-42 model or test predictions.

Only `configs/final_*_clean.yaml` belong to the final protocol. Other configuration files and Colab notes are retained solely as exploratory development history and should not be used to reproduce the submitted results.
