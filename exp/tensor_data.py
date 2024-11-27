import warnings
from typing import List
from typing import Literal
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from numpy.random import Generator
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from pydmdeep.data import generate_toy_dataset
from pydmdeep.types import Float1D
from pydmdeep.types import Float2D
from pydmdeep.types import Float3D
from pydmdeep.types import Int1D

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"


def run(
    seed: int,
    lags: int,
    train_len: float,
    rand: bool,
    scaler: Literal["minmax", "std"],
    target_is_statespace: bool,
    k_modes: Optional[int] = None,
):
    """
    Create train/val/test TensorDataset to be passed through LSTM network.

    Parameters:
    -----------
    seed: randomization seed for reproduciabiilty.
    lags: number of right singular vectors to be used for lstm training input.
    train_len: percentage of data to be used for training.
    rand: shuffle or sequential indices.
    scaler: typing of data scaling to use for input and output data.
    target_is_statespace: True right singular vectors for input and original statespace
                          for creating training data. False uses right singular vectors
                          for both input and output training.
    k_modes: number of right singular vectors to use for training. Selected after
             scaling.
    """
    if DEVICE == "CPU":
        warnings.warn("Using CPU instead of cuda.", stacklevel=2)

    dataset = generate_toy_dataset(tmax=8 * np.pi, nt=129 * 8, nx=65 * 8)
    time_delay1 = dataset["time_delay1"]
    U, S, Vh = np.linalg.svd(time_delay1, full_matrices=False)

    nx, nt = Vh.shape

    if not k_modes:
        k_modes = nx
    if rand:
        rng = np.random.default_rng(seed=seed)
        train_idx, val_idx = _train_val_idxs(nt - lags, train_len=train_len, rng=rng)
    else:
        train_idx, val_idx = _train_val_idxs(nt - lags, train_len=train_len)

    test_idx = val_idx[1::2]
    val_idx = val_idx[::2]
    if scaler == "minmax":
        input_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    elif scaler == "std":
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
    else:
        raise ValueError(f"scaler literal: '{scaler}' not a valid input.")

    input_scaler.fit(Vh.T[train_idx])
    target_scaler.fit(time_delay1.T[train_idx])

    V_scaled = input_scaler.transform(Vh.T)
    # (nt, nx)
    time_delay1_scaled = target_scaler.transform(time_delay1.T)

    if target_is_statespace:
        data_seq_in, data_seq_out = _create_data_seq(
            data_in=V_scaled[:, :k_modes], data_out=time_delay1_scaled, lags=lags
        )
    else:
        data_seq_in, data_seq_out = _create_data_seq(
            data_in=V_scaled[:, :k_modes], lags=lags
        )

    train, val, test = create_tensor_data(
        data_seq_in, data_seq_out, idxs=[train_idx, val_idx, test_idx], device=DEVICE
    )

    results = {
        "tensor_dataset": (train, val, test),
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "k_modes": k_modes,
        "dataset": dataset,
        "lags": lags,
        "target_is_statespace": target_is_statespace,
    }

    explained_variance = S**2 / np.sum(S**2)
    plot_explained_variance(explained_variance)

    return {"main": None, "data": results}


def create_tensor_data(
    in_data: Float2D, out_data: Float2D, idxs: List[Int1D], device: str
) -> List[TensorDataset]:
    """
    Convert data_sequences and target labels to TensorDataset based on desired indices.
    Send data to gpu.

    Parameters:
    ----------
    in_data: input data sequence
    out_data: target_labels
    idxs: List of indices to split into separate Tensordatasets. i.e. train/val/test.
    device: where to store data on. `cuda` or `cpu`.

    Returns:
    -------
    tensor_dataset: list of TensorDatasets to be passed to Dataloaders or torch module.

    """
    tensor_datasets = []
    for idx in idxs:
        in_data_tensor = torch.tensor(in_data[idx], dtype=torch.float32).to(device)
        out_data_tensor = torch.tensor(out_data[idx], dtype=torch.float32).to(device)
        tensor_datasets.append(TensorDataset(in_data_tensor, out_data_tensor))
    return tensor_datasets


def _create_data_seq(
    data_in: Float2D, data_out: Optional[Float2D] = None, lags: int = 1
) -> tuple[Float3D, Float2D]:
    """
    Convert dataset into data sequences (both input and output targets) for LSTM
    learning.

    Parameters:
    ----------
    data_in: matrix containing timeseries (nt,nx_in)
    data_out: Optional data_out mapping (nt, nx_out)
    lags: size of input mapping to be created for data_seq_in

    Returns:
    -------
    data_seq_in: sequence inputs
    data_seq_out: target labels for sequences.
    """
    nt, nx_in = data_in.shape
    if data_out is None:
        data_out = data_in
    nt, nx_out = data_out.shape
    data_seq_in = np.zeros((nt - lags, lags, nx_in))
    data_seq_out = np.zeros(((nt - lags, nx_out)))
    for idx in range(nt - lags):
        data_seq_in[idx] = data_in[idx : idx + lags]
        data_seq_out[idx] = data_out[idx + lags]
    return data_seq_in, data_seq_out


def _train_val_idxs(
    n_len: int, train_len: float, rng: Optional[Generator] = None
) -> tuple[Int1D, Int1D]:
    """
    Create train and val/test indices for n_len indices where train_len indicates
    number of indices to be used for training.
    """
    idxs = np.arange(n_len)
    if rng is not None:
        rng.shuffle(idxs)
    n_train = int(n_len * train_len)
    return idxs[:n_train], idxs[n_train:]


def plot_explained_variance(explained_variance: Float1D) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    explained_variance = np.cumsum(explained_variance)

    ax.scatter(
        range(1, explained_variance.shape[0] + 1),
        explained_variance * 100,
        label="Variance %",
    )

    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Percentage Explained Variance")
    ax.set_title("Percentage of Variance Explained by Singular Values")
    ax.legend()
    ax.grid(True)
    return fig
