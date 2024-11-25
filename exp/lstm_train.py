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
    hidden_size: int,
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
    input_scaler = data["input_scaler"]
    target_scaler = data["target_scaler"]
    k_modes = data["k_modes"]
    dataset = data["dataset"]
    time_delay_test = dataset["time_delay1"]
    target_is_statespace = data["target_is_statespace"]

    U, S, Vh = np.linalg.svd(time_delay_test, full_matrices=False)

    V_scaled = input_scaler.transform(Vh.T)

    # Check if data is on cpu
    if (
        train.tensors[0].device == "CPU"
        or val.tensors[0] == "CPU"
        or test.tensors[0].device == "CPU"
    ):
        warnings.warn("Using CPU instead of cuda.", stacklevel=2)

    # instantiate model
    _, _, d_in = train.tensors[0].shape 
    _, d_out = train.tensors[1].shape

    lstm_model = LSTMModel(
        input_size=d_in,
        hidden_size=hidden_size,
        output_size=d_out,
        num_layers=num_layers,
    ).to(DEVICE)

    opt, opt_kws = opt
    optimizer = opt(lstm_model.parameters(), **opt_kws)
    loss = loss()

    if not model_trainer_kws:
        model_trainer_kws = {"minimum_loss_decrease": 1e-5, "patience": 10}
    
    # does not reconstruct orig statespace in model_trainer
    if target_is_statespace:
        V_test_dataset = None
    else:
        V_test_dataset = V_scaled[:,:k_modes]

    # Train model
    train_dataloader = DataLoader(dataset=train, **dataloader_kws)
    train_losses, reconstruction_errors = model_trainer(
        model=lstm_model,
        epochs=num_epochs,
        dataloader=train_dataloader,
        optimizer=optimizer,
        loss_criterion=loss,
        device=DEVICE,
        lags=lags,
        V_test_dataset=V_test_dataset,
        **model_trainer_kws,
    )

    plot_train_loss(train_losses)

    # returns empty array if target_is_statespace == True
    if not target_is_statespace:
        plot_reconstruction_loss(reconstruction_errors)


    # Reconstruct original statespace
    V_scaled_tensor = torch.Tensor(V_scaled).to(DEVICE)
    lstm_model.eval()
    with torch.no_grad():
        lstm_prediction = construct_prediction(
            V=V_scaled_tensor[:,:k_modes],
            model=lstm_model,
            lags=lags,
            device=DEVICE
        )
    
    if target_is_statespace:
        time_delay_lstm = lstm_prediction
        # Unscale prediction
        time_delay_lstm_np = target_scaler.inverse_transform(
            time_delay_lstm.cpu().numpy()
        )
        # tack on first lags (nt, nx)
        time_delay_lstm_np = np.vstack(
            (time_delay_test.T[:lags,:],time_delay_lstm_np)
        )

    else:
        V_lstm_scaled = lstm_prediction

        # unscale V
        V_lstm_scaled_np = V_lstm_scaled.cpu().numpy()

        inter_mat = np.zeros_like(Vh.T)
        inter_mat[lags:,:k_modes] = V_lstm_scaled_np

        V_lstm_unscaled_np = input_scaler.inverse_transform(inter_mat)
        V_lstm_unscaled_np[:lags,:k_modes] = Vh[:k_modes,:lags].T
        V_lstm_unscaled_np[:,k_modes:] = 0

        time_delay_lstm_np = (U*S)@(V_lstm_unscaled_np.T)
        time_delay_lstm_np = time_delay_lstm_np.T
        
    print(f"target_is_statespace: {target_is_statespace}")
    plot_time_delay_lstm(time_delay_lstm_np)


    results = {
        "train_losses": train_losses,
        "reconstruction": reconstruction_errors,
        "lstm_model": lstm_model,
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

def plot_time_delay_lstm(time_delay: Float2D):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(time_delay, aspect='auto')  
    fig.colorbar(cax, ax=ax, orientation='vertical')  
    ax.set_title("Time Delay Matrix")
    ax.set_xlabel("Features")
    ax.set_ylabel("Time steps")
    plt.tight_layout()  
    plt.show()

def construct_prediction(
        V:torch.Tensor,
        model: LSTMModel, 
        lags:int, 
        device: Literal["cuda"]
)->torch.Tensor:
    nt,_ = V.shape
    nx_out = model.output_size

    pred = torch.zeros((nt-lags,nx_out)).to(device)
    for idx in range(nt-lags):
        pred_idx = model(V[idx:idx+lags].unsqueeze(0))
        pred[idx] = pred_idx
    return pred