import numpy as np
from drx.metrics import expected_calibration_error, summarise


def test_metrics(tmp_path):
    y=np.array([0,1,2,3,4,4,0,2])
    p=np.eye(5)[y]*.85+.03
    result=summarise(y,p,tmp_path)
    assert result['accuracy']==1.0
    assert result['macro_f1']==1.0
    assert (tmp_path/'metrics.json').exists()
    assert (tmp_path/'confusion_matrix.png').exists()
    assert (tmp_path/'reliability.png').exists()
    assert result['undergraded']==0


def test_bad_probabilities():
    import pytest
    with pytest.raises(ValueError):
        expected_calibration_error([0],np.ones((1,5)))
