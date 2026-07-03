import os
from src.train import train


def test_training():
    train()
    assert os.path.exists("models/model.pkl")
