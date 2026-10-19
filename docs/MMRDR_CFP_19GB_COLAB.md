# MMRDR-CFP, 19 GB multipart archive: free Colab workflow

**Purpose:** extract only the CFP modality, produce reliable train/test manifests, and run the frozen-head vs partially fine-tuned ResNet-50 assessment without duplicating the original images.

## 1. What to download and where

- Official dataset: https://figshare.com/articles/dataset/MMRDR/29423747
- The complete roughly **18.6 GB archive** is split into nine volumes: `MMRDR.zip.001` through `MMRDR.zip.009`. These are nine consecutive pieces **of the same ZIP**, not nine independent ZIPs or separate modalities.
- The combined archive has **CFP, OCT, UWF**. Our assessment uses only **CFP**, but you may need all nine compressed volumes to extract that subset.
- **Use Colab's temporary `/content` disk, not your 8 GB MacBook and not an unverified free 15 GB Drive allowance.** Check free space first. Colab runtime and GPU/disk availability vary. You need roughly 19 GB for parts **plus** room for extracted CFP and model outputs. If you lack space, stop and discuss a suitable alternative.
- Do not create a second 19 GB joined archive unless required by the extraction tool and sufficient free disk exists. Do not upload raw MMRDR images to public GitHub.

## 2. Start Colab and inspect resources

Open https://colab.research.google.com and select GPU runtime if available.

```python
!df -h /content
!nvidia-smi
!git clone https://github.com/abhpasha/clinically-explainable-dr.git
%cd /content/clinically-explainable-dr
!pip -q install -r requirements.txt
!pip -q install -e .
!pytest -q
```

## 3. List Figshare files programmatically

Run in Colab; inspect actual reported filenames and sizes before downloading. A Figshare API download link may require redirection or may fail if temporarily rate-limited.

```python
import requests
api = requests.get("https://api.figshare.com/v2/articles/29423747",timeout=60)
api.raise_for_status()
files = api.json()["files"]
for f in files:
    print(f["name"],round(f["size"]/1e9,2),"GB",f["download_url"])
```

Download the nine volumes **directly to** `/content/mmrdr_parts/` using the URLs printed by this cell (not your MacBook browser download). Check part numbering and sizes. One possible Colab downloader, only if the API names match:

```python
import re, subprocess
from pathlib import Path
parts=sorted((f for f in files if re.fullmatch(r"MMRDR\.zip\.\d{3}",f["name"])),key=lambda f:f["name"])
assert len(parts)==9, "Unexpected naming/count: stop and inspect API results"
target=Path("/content/mmrdr_parts"); target.mkdir(exist_ok=True)
for f in parts:
    dest=target/f["name"]
    if dest.exists() and dest.stat().st_size == f["size"]:
        print("Already complete",f["name"]);continue
    subprocess.run(["wget","-c","--tries=3","--timeout=120","-O",str(dest),f["download_url"]],check=True)
    assert dest.stat().st_size==f["size"],f"Incomplete: {f['name']}"
```

Note: if a download interrupts, Colab may lose temporary files after a restart. Resume depends on the download server supporting HTTP byte ranges. If your disk is too small, do not launch this cell.

## 4. Check the archive and extract **only CFP**

```python
!sudo apt-get update -qq
!sudo apt-get install -y -qq p7zip-full
!ls -lh /content/mmrdr_parts/
!7z l /content/mmrdr_parts/MMRDR.zip.001 | tail -35
!df -h /content
```

Make sure that `7z l` succeeds and that the archive paths begin with `MMRDR-CFP/`. The command below is a **first attempt**, not a guarantee for every split-ZIP tool/version. If it fails, stop and send the error; don't delete or recombine anything yet.

```python
!mkdir -p /content/mmrdr_data
!7z x -y /content/mmrdr_parts/MMRDR.zip.001 "-o/content/mmrdr_data" "MMRDR-CFP/*"
!find /content/mmrdr_data -name FP.csv
!find /content/mmrdr_data -name '*.jpg' | wc -l
!df -h /content
```

The paper reports **11,118 CFP images** across the published train/test split. A smaller count indicates partial extraction or wrong paths. If the archive path is nested, adjust after viewing the archive listing. Full ZIP extraction and recompression are unnecessary.

## 5. Audit the real CSV and prepare manifests, with NO image copies

```python
import pandas as pd
from pathlib import Path
root=Path("/content/mmrdr_data/MMRDR-CFP")
print(pd.read_csv(root/"FP.csv").columns.tolist())
print(pd.read_csv(root/"FP.csv").head(3).to_string(index=False))
print(len(list(root.rglob("*.jpg"))))
```

If `FP.csv` has `image` and `grade` with filenames `img/tr000001.jpg` / `img/ts000001.jpg`, run:

```python
!python -m drx.prepare --csv /content/mmrdr_data/MMRDR-CFP/FP.csv --images /content/mmrdr_data/MMRDR-CFP --output data/manifests
!cat data/manifests/audit.json
```

This **does not resize, recompress or duplicate** the images. It splits by the dataset's existing official filename prefixes and checks paths, labels, class counts and a sample of JPEG integrity. Training/validation are subsequently split **inside the official training partition**. This CFP split is at the **image level**, not proven patient-independent.

## 6. Run the experiments (preprocessing occurs during each mini-batch)

```python
!python -m drx.train --config configs/resnet50_head.yaml --train-csv data/manifests/train.csv --train-images /content/mmrdr_data/MMRDR-CFP
!python -m drx.train --config configs/resnet50_partial.yaml --train-csv data/manifests/train.csv --train-images /content/mmrdr_data/MMRDR-CFP
```

**Head-only:** pretrained ImageNet ResNet-50, frozen backbone, trained five-grade head at 224 px. **Partial:** start from the best frozen-head checkpoint and update `layer4` + `fc`, keeping the rest frozen. Both use the same seeded train/validation split; select final checkpoint using validation macro-F1 **before** inspecting official test labels. Optional 384 px third stage: `configs/resnet50_partial_384.yaml`; first confirm Colab GPU memory/time.

The data loader automatically resizes JPEGs and normalises channels when training batches are loaded, and applies mild random augmentation only to training images. Original images remain unmodified. Do **not** pre-resize everything to 224; small DR lesions can disappear at low resolution, so 384 px is valuable only if compute allows.

## 7. Evaluate and back up evidence

After model selection, evaluate once on **official** `test.csv`:

```python
!python -m drx.evaluate --checkpoint outputs/resnet50_partial/best.pt --test-csv data/manifests/test.csv --test-images /content/mmrdr_data/MMRDR-CFP --output-dir outputs/test
```

**Important:** Replace checkpoint with the validation-selected model if head-only performed better; report all comparisons honestly. Then use `python -m drx.explain` with actual test CFP image paths to generate Grad-CAM examples. Copy results/selected checkpoints to your own persistent Drive space after checking quota; Colab local disk disappears after runtime reset. Never publish the 19 GB dataset or patient-identifiable material.

We have not trained or evaluated the model on actual CFP data yet. Code tests only check the pipeline's software behaviour.
