import argparse
from pathlib import Path
import pandas as pd
from .data import parse_manifest, validate_no_overlap


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-csv",required=True)
    p.add_argument("--train-images",required=True)
    p.add_argument("--test-csv",required=True)
    p.add_argument("--test-images",required=True)
    p.add_argument("--filename-col")
    p.add_argument("--label-col",default="grade")
    a=p.parse_args()
    tr=parse_manifest(a.train_csv,a.train_images,a.filename_col,a.label_col)
    te=parse_manifest(a.test_csv,a.test_images,a.filename_col,a.label_col)
    validate_no_overlap(tr,te)
    print("OFFICIAL TRAIN counts:",tr.grade.value_counts().sort_index().to_dict())
    print("OFFICIAL TEST counts:",te.grade.value_counts().sort_index().to_dict())
    print("No exact path or filename collisions. Patient-level independence is NOT established.")
if __name__=="__main__":main()
