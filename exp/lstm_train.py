import warnings
from typing import Any
from typing import Optional
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from pydmdeep.models.lstm import LSTMModel, model_trainer, reconstruct_V
from pydmdeep.types import Float1D
from pydmdeep.types import Float2D

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"


def run(
    data: dict[Any],
    opt: tuple[torch.optim.Optimizer, dict[Any]],
    loss: nn.Module,
    dataloader_kws: dict[Any],
    num_layers: int,
    num_epochs: int,
    seed: int,
    model_trainer_kws: Optional[dict] = None,
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

    U, S, Vh = np.linalg.svd(time_delay_test, full_matrices=False)

    V_scaled = data_transformer.transform(Vh.T)

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

    if not model_trainer_kws:
        model_trainer_kws = {"minimum_loss_decrease": 1e-5, "patience": 10}

    train_dataloader = DataLoader(dataset=train, **dataloader_kws)
    train_losses, reconstruction_errors = model_trainer(
        model=lstm_model,
        epochs=num_epochs,
        dataloader=train_dataloader,
        optimizer=optimizer,
        loss_criterion=loss,
        device=DEVICE,
        lags=lags,
        V_test_dataset=V_scaled,
        **model_trainer_kws,
    )

    plot_train_loss(train_losses)
    plot_reconstruction_loss(reconstruction_errors)

    V_scaled_tensor = torch.Tensor(V_scaled).to(DEVICE)
    lstm_model.eval()
    with torch.no_grad():
        V_lstm_scaled = construct_Vlstm(
            V=V_scaled_tensor, 
            model=lstm_model,
            lags=lags,
            device=DEVICE
        )
        Vhat, V = reconstruct_V(
            V_scaled=V_scaled_tensor,
            model=lstm_model,
            t_len=len(V_scaled_tensor),
            lags=lags,
            device=DEVICE)
    
    def torch_diff(A,B):
        return torch.linalg.norm(A-B,"fro")/ torch.linalg.norm(A,"fro")

    print("Ike:", torch_diff(V,V_lstm_scaled))
    print("Johnson:", torch_diff(V,Vhat))


    V_lstm_np = data_transformer.inverse_transform(V_lstm_scaled.cpu().numpy())
    Vhat_np = data_transformer.inverse_transform(Vhat.cpu().numpy())

    def np_diff(A,B):
        return np.linalg.norm(A-B)/np.linalg.norm(A)
    
    print("Ike2:", np_diff(Vh.T, V_lstm_np))
    print("Johnson2:", np_diff(Vh.T, Vhat_np))

    time_delay_lstm = (U*S)@(V_lstm_np.T)

    plot_time_delay_lstm(time_delay_lstm)


    results = {
        "train_losses": train_losses,
        "reconstruction": reconstruction_errors,
        "lstm_mode": lstm_model,
        "prev_data": data
    }
    

    return {"main": (train_losses[-1],reconstruction_errors[-1]), "data": results}


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

def construct_Vlstm(
        V: torch.Tensor, model: LSTMModel,lags:int ,device: Literal["cuda"]
)->torch.Tensor:
    nx, nt = V.shape
    V_lstm = torch.zeros((nx,nt)).to(device)
    V_lstm[:lags] = V[:lags]
    for idx in range(nt-lags):
        pred = model(V[idx:idx+lags].unsqueeze(0))
        V_lstm[idx:idx+lags] = pred
    return V_lstm


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_time_delay_lstm(time_delay: Float2D):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(time_delay, aspect='auto')  
    fig.colorbar(cax, ax=ax, orientation='vertical')  
    ax.set_title("Time Delay Matrix")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Features")
    plt.tight_layout()  
    plt.show()
