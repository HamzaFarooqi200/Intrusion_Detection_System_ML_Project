# tests/test_data.py

from src.train import load_data


def test_data_loading():
    X, y = load_data("data/kddcup.data_10_percent.gz")
    assert X.shape[0] > 0
    assert len(X) == len(y)