"""Deterministic exact-pixel deduplication and fixed CFP development/validation split."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def _resolve(images,name):
    images=Path(images); name=Path(str(name)).name
    c=[images/name,images/"img"/name]
    if images.name.lower()=="img": c.append(images.parent/name)
    f=list(dict.fromkeys(p.resolve() for p in c if p.is_file()))
    if len(f)!=1: raise FileNotFoundError(f"Expected one image for {name!r}; found {len(f)}")
    return f[0]

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def sha256_pixels(path):
    with Image.open(path) as im:
        a=np.asarray(im.convert("RGB"),dtype=np.uint8)
    h=hashlib.sha256(); h.update(str(a.shape).encode("ascii")); h.update(a.tobytes())
    return h.hexdigest()

def prepare(csv,images,output,val_fraction=.15,seed=42):
    src=pd.read_csv(csv).copy()
    if not {"image","grade"}.issubset(src.columns): raise ValueError("FP.csv requires image and grade")
    src["image"]=src.image.astype(str).map(lambda x:Path(x).name)
    src["grade"]=pd.to_numeric(src.grade,errors="raise").astype(int)
    names=src.image.str.lower()
    tr=names.str.fullmatch(r"tr\d{6}\.jpe?g"); ts=names.str.fullmatch(r"ts\d{6}\.jpe?g")
    if not (tr|ts).all() or not tr.any() or not ts.any(): raise ValueError("Unexpected CFP naming")
    rows=[]
    for subset,mask in (("official_train",tr),("official_test",ts)):
        for r in src.loc[mask,["image","grade"]].itertuples(index=False):
            p=_resolve(images,r.image)
            rows.append({"subset":subset,"image":r.image,"grade":int(r.grade),
                         "file_sha256":sha256_file(p),"pixel_sha256":sha256_pixels(p)})
    audit=pd.DataFrame(rows)
    train=audit[audit.subset=="official_train"].copy()
    test=audit[audit.subset=="official_test"].copy()
    conflicts=train.groupby("pixel_sha256").grade.nunique()
    bad=set(conflicts[conflicts>1].index)
    conflict_rows=train[train.pixel_sha256.isin(bad)].copy()
    ok=train[~train.pixel_sha256.isin(bad)].sort_values(["pixel_sha256","image"])
    keep=ok.groupby("pixel_sha256",as_index=False).head(1).copy()
    dup=ok[~ok.index.isin(keep.index)].copy()
    th=set(test.pixel_sha256)
    overlap=keep[keep.pixel_sha256.isin(th)].copy()
    pool=keep[~keep.pixel_sha256.isin(th)].copy()
    dev,val=train_test_split(pool,test_size=val_fraction,random_state=seed,stratify=pool.grade)
    dev=dev.sort_values("image").reset_index(drop=True); val=val.sort_values("image").reset_index(drop=True)
    assert not (set(dev.pixel_sha256)&set(val.pixel_sha256))
    assert not (set(dev.pixel_sha256)&set(test.pixel_sha256))
    assert not (set(val.pixel_sha256)&set(test.pixel_sha256))
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    audit.to_csv(out/"all_official_hashes.csv",index=False)
    conflict_rows.to_csv(out/"conflicting_exact_pixel_duplicates.csv",index=False)
    dup.to_csv(out/"removed_same_label_exact_pixel_duplicates.csv",index=False)
    overlap.to_csv(out/"removed_official_train_test_exact_pixel_overlaps.csv",index=False)
    dev[["image","grade"]].to_csv(out/"development.csv",index=False)
    val[["image","grade"]].to_csv(out/"validation.csv",index=False)
    test[["image","grade"]].to_csv(out/"official_test.csv",index=False)
    summary={"official_train_original":int(len(train)),"official_test":int(len(test)),
             "conflicting_label_groups_excluded":int(len(bad)),
             "conflicting_label_images_excluded":int(len(conflict_rows)),
             "same_label_exact_pixel_duplicates_removed":int(len(dup)),
             "official_train_images_removed_for_exact_test_pixel_overlap":int(len(overlap)),
             "clean_pool_before_internal_split":int(len(pool)),"development":int(len(dev)),
             "validation":int(len(val)),"seed":seed,"val_fraction":val_fraction}
    (out/"clean_split_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    import argparse
    p=argparse.ArgumentParser(description="Create leakage-controlled MMRDR-CFP development/validation/test manifests.")
    p.add_argument("--csv",required=True,help="Path to MMRDR-CFP FP.csv")
    p.add_argument("--images",required=True,help="Path to MMRDR-CFP directory or img/ directory")
    p.add_argument("--output",required=True,help="Output directory for manifests and hash audit")
    p.add_argument("--val-fraction",type=float,default=0.15)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    print(json.dumps(prepare(a.csv,a.images,a.output,a.val_fraction,a.seed),indent=2))

if __name__=="__main__":
    main()
