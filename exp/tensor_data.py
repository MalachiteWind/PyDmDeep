from pydmdeep.data import generate_toy_dataset, create_tensor_data
import warnings
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
import numpy as np
import torch
from pydmdeep.types import Float2D, Int1D

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"



def run(seed:int):
    # rng = np.random.default_rng(seed=seed)
    dataset = generate_toy_dataset(tmax=8*np.pi,nt =129*8,nx= 65*8)
    U,S,Vh = np.linalg.svd(dataset["time_delay1"])

    train, val, test = create_tensor_data(
        data_mat=Vh,
        lags=32,
        train_len=0.9,
        device=DEVICE

    )
    
    return train,val,test


def _train_val_idxs(n_len: int, train_len: float,) -> tuple[Int1D, Int1D]:
    idxs = np.arange(n_len)
    np.random.shuffle(idxs)
    n_train = int(n_len * train_len)
    return idxs[:n_train], idxs[n_train:]

def create_tensor_data(
    data_mat: Float2D, lags: int, train_len: float, device: str = "cpu"
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    

    Parameters:
    ----------
    data_mat: shape (nt,nx) samples are the rows.
    """
    nt, nx = data_mat.shape
    train_idxs, val_idxs = _train_val_idxs(nt - lags, train_len=train_len)
    test_idxs = val_idxs[1::2]
    val_idxs = val_idxs[::2]

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(data_mat[train_idxs])

    transformed_data = min_max_scaler.transform(data_mat)

    nt, nx = data_mat.shape

    data_seq = np.zeros((nt - lags, lags, nx))
    for idx in range(nt - lags):
        data_seq[idx] = transformed_data[idx : idx + lags]

    train_data_in = torch.tensor(data=data_seq[train_idxs], dtype=torch.float32).to(device)
    val_data_in = torch.tensor(data=data_seq[val_idxs], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(data=data_seq[test_idxs], dtype=torch.float32).to(device)

    
    if device == "cpu":
        warnings.warn("CPU used instead of GPU.", stacklevel=2)

    train_data_out = torch.tensor(
        transformed_data[train_idxs + lags - 1], dtype=torch.float32
    ).to(device)
    val_data_out = torch.tensor(
        transformed_data[val_idxs + lags - 1], dtype=torch.float32
    ).to(device)
    test_data_out = torch.tensor(
        transformed_data[test_idxs + lags - 1], dtype=torch.float32
    ).to(device)

    train_data = TensorDataset(train_data_in, train_data_out)
    val_data = TensorDataset(val_data_in, val_data_out)
    test_data = TensorDataset(test_data_in, test_data_out)

    return train_data, val_data, test_data, min_max_scaler

