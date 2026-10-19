import pytest
import pandas as pd
from PIL import Image
from drx.data import parse_manifest, make_split, validate_no_overlap


def test_csv_paths_and_split(tmp_path):
    paths=[];grades=[]
    for grade in range(5):
        for j in range(8):
            name=f"tr{grade:01d}{j:05d}.jpg"
            Image.new("RGB",(24,24),color=(grade*40,100,30)).save(tmp_path/name)
            paths.append(name);grades.append(grade)
    csv=tmp_path/'FP.csv';pd.DataFrame({'image':paths,'grade':grades}).to_csv(csv,index=False)
    df=parse_manifest(csv,tmp_path)
    a,b=make_split(df,.25,42)
    assert len(a)+len(b)==40
    assert set(a.path).isdisjoint(b.path)
    assert set(b.grade)==set(range(5))
    with pytest.raises(ValueError,match='LEAKAGE'):
        validate_no_overlap(df,df)


def test_invalid_grade(tmp_path):
    Image.new("RGB",(20,20)).save(tmp_path/'a.jpg')
    csv=tmp_path/'FP.csv';pd.DataFrame({'image':['a.jpg'],'grade':[5]}).to_csv(csv,index=False)
    with pytest.raises(ValueError,match='grade'):
        parse_manifest(csv,tmp_path)


def test_manifest_img_prefixed_path(tmp_path):
    (tmp_path / "img").mkdir()
    Image.new("RGB", (20,20)).save(tmp_path / "img" / "tr000001.jpg")
    csv = tmp_path / "FP.csv"
    pd.DataFrame({"image": ["img/tr000001.jpg"], "grade": [2]}).to_csv(csv, index=False)
    for root in [tmp_path, tmp_path / "img"]:
        result = parse_manifest(csv, root)
        assert result.grade.tolist() == [2]
        assert result.path.iloc[0] == str((tmp_path / "img" / "tr000001.jpg").resolve())
