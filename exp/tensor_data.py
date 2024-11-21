import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from numpy.random import Generator
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

from pydmdeep.data import generate_toy_dataset
from pydmdeep.types import Float1D, Float2D, Float3D, Int1D

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"


def run(seed: int, lags: int, train_len: float):
    """
    Create train/val/test TensorDataset to be passed through LSTM network.

    Parameters:
    -----------
    seed: randomization seed for reproduciabiilty.
    lags: number of right singular vectors to be used for lstm training input.
    train_len: percentage of data to be used for training.
    """
    rng = np.random.default_rng(seed=seed)
    if DEVICE == "CPU":
        warnings.warn("Using CPU instead of cuda.", stacklevel=2)

    dataset = generate_toy_dataset(tmax=8 * np.pi, nt=129 * 8, nx=65 * 8)
    U, S, Vh = np.linalg.svd(dataset["time_delay1"])

    nt, nx = Vh.shape

    train_idx, val_idx = _train_val_idxs(nt - lags, train_len=train_len, rng=rng)
    test_idx = val_idx[1::2]
    val_idx = val_idx[::2]

    min_max_scalaer = MinMaxScaler()
    min_max_scalaer.fit(Vh[train_idx])

    Vh_transformed = min_max_scalaer.transform(Vh)

    data_seq_in, data_seq_out = _create_data_seq(Vh_transformed, lags=lags)

    train, val, test = create_tensor_data(
        data_seq_in, data_seq_out, idxs=[train_idx, val_idx, test_idx], device=DEVICE
    )

    results = {
        "tensor_dataset": (train, val, test),
        "transformer": min_max_scalaer,
        "dataset": dataset,
        "lags": lags,
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


def _create_data_seq(data: Float2D, lags: int) -> tuple[Float3D, Float2D]:
    """
    Convert dataset into data sequences (both input and output targets) for LSTM
    learning.

    Parameters:
    ----------
    data: matrix containing timeseries (nt,nx)
    lags: size of input mapping to be created for data_seq_in

    Returns:
    -------
    data_seq_in: sequence inputs
    data_seq_out: target labels for sequences.
    """
    nt, nx = data.shape
    data_seq_in = np.zeros((nt - lags, lags, nx))
    data_seq_out = np.zeros(((nt - lags, nx)))
    for idx in range(nt - lags):
        data_seq_in[idx] = data[idx : idx + lags]
        data_seq_out[idx] = data[idx + lags]
    return data_seq_in, data_seq_out


def _train_val_idxs(
    n_len: int, train_len: float, rng: Generator
) -> tuple[Int1D, Int1D]:
    """
    Create train and val/test indices for n_len indices where train_len indicates
    number of indices to be used for training.
    """
    idxs = np.arange(n_len)
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
