import pandas as pd
import pytest
from PIL import Image
from drx.prepare import prepare
from drx.data import parse_manifest, validate_no_overlap

def test_prepare_from_official_combined_csv(tmp_path):
    root=tmp_path/"MMRDR-CFP"
    (root/"img").mkdir(parents=True)
    rows=[]
    for prefix in ("tr","ts"):
        for n in range(1,11):
            name=f"{prefix}{n:06d}.jpg"
            Image.new("RGB",(20,20),color=(100,50,30)).save(root/"img"/name)
            rows.append({"image":f"img/{name}","grade":(n-1)%5,"lesion":"[0,0,0,0,0,0,0]"})
    pd.DataFrame(rows).to_csv(root/"FP.csv",index=False)
    info=prepare(root/"FP.csv",root,tmp_path/"manifests",2)
    assert info["official_train"]==10 and info["official_test"]==10
    train=parse_manifest(tmp_path/"manifests"/"train.csv",root)
    test=parse_manifest(tmp_path/"manifests"/"test.csv",root)
    assert len(train)==len(test)==10
    validate_no_overlap(train,test)

def test_prepare_incomplete_archive(tmp_path):
    (tmp_path/"img").mkdir()
    Image.new("RGB",(20,20)).save(tmp_path/"img"/"tr000001.jpg")
    pd.DataFrame([{"image":"img/tr000001.jpg","grade":0}]).to_csv(tmp_path/"FP.csv",index=False)
    with pytest.raises(ValueError,match="BOTH"):
        prepare(tmp_path/"FP.csv",tmp_path,tmp_path/"out")
