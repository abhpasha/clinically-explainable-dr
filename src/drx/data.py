"""MMRDR CFP manifests. Never infer test labels from training or vice versa."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T

from .core import N_CLASSES

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_manifest(csv_file, images_dir, filename_col=None, label_col="grade"):
    """Read original FP.csv or normalised manifest; return absolute paths and grades.

    Original dataset fields may vary. Explicitly set --filename-col/--label-col
    whenever auto detection would be ambiguous. We refuse guesses for labels.
    """
    csv_file, images_dir = Path(csv_file), Path(images_dir)
    df = pd.read_csv(csv_file)
    df.columns = [str(c).strip() for c in df.columns]
    if label_col not in df.columns:
        possible = [c for c in df.columns if c.lower() in {"grade", "dr_grade", "dr", "label"}]
        if len(possible) != 1:
            raise ValueError(f"Missing label column '{label_col}'. CSV columns: {list(df.columns)}; supply --label-col")
        label_col = possible[0]
    if filename_col is None:
        possible = [c for c in df.columns if c.lower() in {"img", "image", "filename", "file", "image_name", "img_name", "name", "id"}]
        if len(possible) != 1:
            raise ValueError(f"Cannot unambiguously find filename column in {list(df.columns)}; supply --filename-col")
        filename_col = possible[0]
    if filename_col not in df.columns:
        raise ValueError(f"Missing filename column {filename_col!r}; columns: {list(df.columns)}")
    if df[label_col].isna().any():
        raise ValueError("Missing DR grades")
    numeric = pd.to_numeric(df[label_col], errors="raise")
    if not np.all(np.equal(numeric, np.floor(numeric))) or not numeric.between(0, N_CLASSES-1).all():
        raise ValueError("DR grade must be an integer in [0,4]")
    labels = numeric.astype(int)

    def resolve(v):
        v = str(v).strip()
        if not v or v == "nan":
            raise ValueError("Empty image filename")
        raw = Path(v)
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError("Unsafe image filename (absolute or path traversal)")
        # Some MMRDR manifests use names like tr000001.jpg; others use
        # img/tr000001.jpg. Accept either an img/ root or its parent.
        roots = [images_dir]
        if images_dir.name.lower() == "img":
            roots.append(images_dir.parent)
        else:
            roots.append(images_dir / "img")
        options = [root / raw for root in roots]
        if not raw.suffix:
            options += [root / (v + ext) for root in roots for ext in (".jpg", ".jpeg", ".png")]
        existing = list(dict.fromkeys(x.resolve() for x in options if x.is_file()))
        if len(existing) != 1:
            raise FileNotFoundError(f"Expected exactly one image for {v!r}; found {len(existing)} under {images_dir}")
        if existing[0].suffix.lower() not in IMAGE_EXTS:
            raise ValueError("Unsupported image format")
        return str(existing[0])

    out = pd.DataFrame({"path": [resolve(v) for v in df[filename_col]], "grade": labels})
    if out.path.duplicated().any():
        raise ValueError("Duplicate image path in manifest")
    return out


def validate_no_overlap(train_df, test_df):
    overlap = set(train_df.path) & set(test_df.path)
    if overlap:
        raise ValueError(f"TRAIN/TEST IMAGE PATH LEAKAGE: {len(overlap)} overlapping files")
    # Filenames may be reused in datasets; reject to be conservative, but not a
    # patient identity or perceptual duplicate guarantee.
    basenames = set(Path(p).name for p in train_df.path) & set(Path(p).name for p in test_df.path)
    if basenames:
        raise ValueError(f"Train/test filename collision: {len(basenames)}")


def make_split(df, val_fraction, seed):
    counts = df.grade.value_counts()
    if counts.min() < 2:
        raise ValueError("At least two examples per grade required for stratified split")
    a, b = train_test_split(df, test_size=val_fraction, random_state=seed, stratify=df.grade)
    return a.sort_index().reset_index(drop=True), b.sort_index().reset_index(drop=True)


def transform(size, training=False):
    aug = [T.Resize((size, size))]
    if training:
        aug += [T.RandomHorizontalFlip(p=.5), T.RandomRotation(10)]
    aug += [T.ToTensor(), T.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225))]
    return T.Compose(aug)


class CFPDataset(Dataset):
    def __init__(self, df, size, training=False):
        self.df = df.reset_index(drop=True)
        self.tf = transform(size, training)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(row.path) as im:
            x = self.tf(im.convert("RGB"))
        return x, int(row.grade), row.path
