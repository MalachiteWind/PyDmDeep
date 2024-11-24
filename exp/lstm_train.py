import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from pydmdeep.models.lstm import LSTMModel, model_trainer
from pydmdeep.types import Float1D

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"


def run(
    data: dict,
    opt: tuple[torch.optim.Optimizer, dict[Any]],
    loss: nn.Module,
    dataloader_kws: dict[Any],
    num_layers: int,
    num_epochs: int,
    seed: int,
):
    """
    Train and create LSTM model on right singular vectors of time delay matrix.

    Parameters:
    -----------
    data: arguments to be passed in from exp.tensor_data step.
    opt: pytorch optimizer and accompanying keyword arguments.
        i.e. (optim.SGD, {"lr":0.1})
    loss: loss_criterion on which to update network weights
    dataloader_kws: batch_size and other keyword arguments for pytorch DataLoader.
    num_layers: number of hidden layer for LSTM model
    num_epochs: number of maximum training epochs (unless early stopage)
    seed: randomzation seed to be set for reproducibility.
    """
    # set seed
    set_seed(seed=seed)

    lags = data["lags"]

    # Load data
    train, val, test = data["tensor_dataset"]
    data_transformer = data["transformer"]
    dataset = data["dataset"]
    time_delay_test = dataset["time_delay1"]

    if (
        train.tensors[0].device == "CPU"
        or val.tensors[0] == "CPU"
        or test.tensors[0].device == "CPU"
    ):
        warnings.warn("Using CPU instead of cuda.", stacklevel=2)

    # instantiate model
    _, d_len = train.tensors[1].shape
    lstm_model = LSTMModel(
        input_size=d_len,
        hidden_size=d_len,
        output_size=d_len,
        num_layers=num_layers,
    ).to(DEVICE)

    opt, opt_kws = opt
    optimizer = opt(lstm_model.parameters(), **opt_kws)
    loss = loss()

    train_dataloader = DataLoader(dataset=train, **dataloader_kws)
    train_losses, reconstruction_errors = model_trainer(
        model=lstm_model,
        epochs=num_epochs,
        dataloader=train_dataloader,
        optimizer=optimizer,
        loss_criterion=loss,
        device=DEVICE,
        lags=lags,
        epoch_test_dataset=time_delay_test,
    )

    plot_train_loss(train_losses)
    plot_reconstruction_loss(reconstruction_errors)

    results = {
        "train_losses": train_losses,
        "reconstruction": reconstruction_errors,
        "lstm_mode": lstm_model,
        "prev_data": data
    }
    

    return {"main": train_losses[-1], "data": results}


def plot_train_loss(train_loss: Float1D) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(train_loss, label="loss", color="b", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True)
    return fig

def plot_reconstruction_loss(reconstruction_loss: Float1D) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(reconstruction_loss, label="error", color="b", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_title("Reconstruction Error")
    ax.legend()
    ax.grid(True)
    return fig


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
