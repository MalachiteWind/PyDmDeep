import warnings

import torch.nn as nn
from torch.utils.data import DataLoader

from pydmdeep.models.lstm import LSTMModel


def run(data: dict, loss: str):

    train, val, test = data["main"]
    min_max_scaler = data["transformer"]
    dataset = data["dataset"]

    if (
        train.tensor[0].device == "CPU"
        or val.tensor[0] == "CPU"
        or test.tesnor[0].device == "CPU"
    ):
        warnings.warn("Using CPU instead of cuda.", stacklevel=2)

    LSTMModel()

    return {"main": 1}
