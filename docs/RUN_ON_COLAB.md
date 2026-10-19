# Run the MMRDR-CFP assessment on a GPU (Google Colab)

**Status:** Runbook, not a record of training. Do not report model scores until the commands have successfully run on real MMRDR-CFP data.

## 1. Acquire the data

- Official dataset: https://figshare.com/articles/dataset/MMRDR/29423747
- Use only the **colour fundus photograph / MMRDR-CFP** subset for this assessment.
- If the download bundles OCT and UWF as well, those are not needed for the five-grade CFP experiment.
- Follow the dataset terms. Keep raw images and personally identifiable information out of public GitHub.
- Dataset source may contain folders named `training`/`testing`, `train`/`test` or other conventions. Inspect the archive; do not assume our example paths match yours.

For uninterrupted access, save the extracted dataset in your own Google Drive. A Colab ephemeral disk is cleared when the runtime resets.

## 2. Start Colab with GPU

Open https://colab.research.google.com/ and choose **Runtime → Change runtime type → GPU**. T4 may be available depending on your allocation; availability is not guaranteed.

Create the first two code cells:

~~~python
from google.colab import drive
drive.mount('/content/drive')
~~~

~~~python
!git clone https://github.com/abhpasha/clinically-explainable-dr.git
%cd /content/clinically-explainable-dr
!pip install -q -r requirements.txt
!pip install -q -e .
!python -m pytest -q
~~~

Check the filesystem and label column names before training:

~~~python
import pandas as pd
from pathlib import Path
DATA = Path("/content/drive/MyDrive/MMRDR-CFP")   # EDIT this to your real directory
print([str(p) for p in DATA.rglob("FP.csv")])
print([str(p) for p in DATA.rglob("*.jpg")][:5])
for csv in DATA.rglob("FP.csv"):
    df = pd.read_csv(csv)
    print(csv, list(df.columns), df.head(3).to_string(index=False))
~~~

**If no files are shown:** Extract the official dataset archive into Drive or the Colab runtime and correct `DATA`. Dataset archive size and extraction time may exceed a free runtime's limits.

## 3. Validate data

Adjust each path to the folders printed in step 2.

~~~bash
python -m drx.inspect \
  --train-csv /content/drive/MyDrive/MMRDR-CFP/train/FP.csv \
  --train-images /content/drive/MyDrive/MMRDR-CFP/train/img \
  --test-csv /content/drive/MyDrive/MMRDR-CFP/test/FP.csv \
  --test-images /content/drive/MyDrive/MMRDR-CFP/test/img \
  --filename-col image --label-col grade
~~~

The loader supports `tr000001.jpg` or `img/tr000001.jpg` filename cells. Use `--filename-col` to match the actual CSV, and confirm **grades are 0 to 4**. File-level separation cannot prove patient-level separation, because the published CFP split lacks patient IDs.

## 4. Train the classifier

~~~bash
python -m drx.train \
  --config configs/resnet50.yaml \
  --train-csv /content/drive/MyDrive/MMRDR-CFP/train/FP.csv \
  --train-images /content/drive/MyDrive/MMRDR-CFP/train/img \
  --test-images /content/drive/MyDrive/MMRDR-CFP/test/img \
  --filename-col image --label-col grade
~~~

The model selects its best checkpoint using **validation macro-F1**, not test labels. The saved checkpoint is `outputs/resnet50/best.pt`. If the GPU runs out of memory, lower batch_size from 16 to 8 or 4 in `configs/resnet50.yaml`. Keep note of every change in the report. A `quick_cpu.yaml` smoke run is not evidence of real performance.

**Before running the held-out test:** inspect `outputs/resnet50/history.csv` and choose/freeze the model based on validation only.

## 5. Evaluate exactly once on the held-out test

~~~bash
python -m drx.evaluate \
  --checkpoint outputs/resnet50/best.pt \
  --test-csv /content/drive/MyDrive/MMRDR-CFP/test/FP.csv \
  --test-images /content/drive/MyDrive/MMRDR-CFP/test/img \
  --filename-col image --label-col grade \
  --output-dir outputs/test
~~~

Inspect `metrics.json`, `predictions.csv`, `confusion_matrix.png` and `reliability.png`. Record per-class recall, grade 3 and 4 underestimation, ECE and Brier score. Distinguish results from the dataset paper's benchmark results.

## 6. Generate honest Grad-CAM case studies

Use real file paths from `outputs/test/predictions.csv`, including correct classifications, severity-undergraded errors and confident errors (if any). Save **4–8** overlays, e.g.:

~~~bash
python -m drx.explain \
  --checkpoint outputs/resnet50/best.pt \
  --image /content/drive/MyDrive/MMRDR-CFP/test/img/ts000001.jpg \
  --output outputs/test/case_01_gradcam.png
~~~

Grad-CAM is attribution for model prediction, not precise disease/lesion segmentation.

## 7. Prepare the report

Fill `report/REPORT_TEMPLATE.md` (max 2,000 words excluding references) from genuine artifacts only. Explain dataset class distribution, image-level leakage limitation, preprocessing, model/validation choices, per-class errors, under-grading, uncertainty and selected explanations. Cite MMRDR and ResNet/Grad-CAM appropriately.

## 8. Export evidence for review

~~~python
import shutil
shutil.make_archive('/content/drive/MyDrive/dr_assessment_results', 'zip',
                    root_dir='/content/clinically-explainable-dr', base_dir='outputs')
~~~

The results bundle contains images/plots and may contain large checkpoints; **avoid putting it in public GitHub**, and check dataset and any patient-image usage restrictions. If sharing for a review, omit raw image data and model checkpoints as appropriate. The report can be revised using `metrics.json`, `history.csv`, confusion and reliability charts, and *permitted* example overlays.

### Completion gate

- [ ] Real MMRDR-CFP downloaded; original CSV columns confirmed
- [ ] Real class frequencies and split audit recorded
- [ ] Full model trained and best checkpoint selected on validation
- [ ] Official held-out test evaluation performed once
- [ ] Grad-CAM examples drawn from genuine images and errors
- [ ] Report includes numeric results, limitations, citations and a proposed challenge mitigation
- [ ] Final deliverables inspected; no private data or checkpoints exposed
