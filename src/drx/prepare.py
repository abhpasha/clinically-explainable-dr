"""Build official MMRDR-CFP train/test manifests without duplicating image data."""
import argparse
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from .data import parse_manifest, validate_no_overlap

def prepare(csv, images, output, sample=20):
    csv, images, output = Path(csv), Path(images), Path(output)
    source = pd.read_csv(csv)
    if not {"image", "grade"}.issubset(source.columns):
        raise ValueError("MMRDR-CFP FP.csv must contain image and grade columns")
    names = source["image"].astype(str).map(lambda x: Path(x).name.lower())
    tr = names.str.fullmatch(r"tr\d{6}\.jpe?g")
    ts = names.str.fullmatch(r"ts\d{6}\.jpe?g")
    if not (tr | ts).all() or tr.sum() == 0 or ts.sum() == 0:
        raise ValueError("Expected BOTH tr######.jpg and ts######.jpg; inspect extracted dataset")
    if names.duplicated().any():
        raise ValueError("Duplicate image names in the official CFP CSV")
    # Verify real paths, grades, and train-test separation before saving anything.
    out_train, out_test = source.loc[tr].copy(), source.loc[ts].copy()
    # Use temp in-memory frame for direct checks of real JPEG references.
    root = images
    def resolve(name):
        v = Path(str(name))
        if v.is_absolute() or ".." in v.parts:
            raise ValueError("Unsafe image path")
        candidates=[root/v,root/"img"/v]
        if root.name.lower()=="img":
            candidates.append(root.parent/v)
        found=list(dict.fromkeys(c.resolve() for c in candidates if c.is_file()))
        if len(found)!=1:
            raise FileNotFoundError(f"Missing or ambiguous image: {name}")
        return found[0]
    train_paths=[resolve(x) for x in out_train["image"]]
    test_paths=[resolve(x) for x in out_test["image"]]
    if set(train_paths) & set(test_paths):
        raise ValueError("Train/test file path overlap")
    for subset in (out_train,out_test):
        grade=pd.to_numeric(subset["grade"],errors="raise")
        if not grade.between(0,4).all() or not (grade == grade.astype(int)).all():
            raise ValueError("DR grade must be integer 0 through 4")
    for path in (train_paths[:max(sample,0)] + test_paths[:max(sample,0)]):
        with Image.open(path) as im:
            im.verify()
    output.mkdir(parents=True,exist_ok=True)
    out_train.to_csv(output/"train.csv",index=False)
    out_test.to_csv(output/"test.csv",index=False)
    summary={"official_train":len(out_train),"official_test":len(out_test),
             "train_grades":{str(k):int(v) for k,v in out_train.grade.value_counts().sort_index().items()},
             "test_grades":{str(k):int(v) for k,v in out_test.grade.value_counts().sort_index().items()},
             "source":str(csv),"image_root":str(root),
             "note":"MMRDR-CFP official split is image-level, not proven patient-independent"}
    (output/"audit.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True,help="Original MMRDR-CFP/FP.csv")
    p.add_argument("--images",required=True,help="MMRDR-CFP root or img directory")
    p.add_argument("--output",default="data/manifests")
    p.add_argument("--sample",type=int,default=20)
    a=p.parse_args()
    print(json.dumps(prepare(a.csv,a.images,a.output,a.sample),indent=2))

if __name__=="__main__":
    main()
