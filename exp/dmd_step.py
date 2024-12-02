from typing import cast
from typing import Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

import matplotlib.pyplot as plt
from pydmd import DMD
from pydmd.preprocessing import hankel_preprocessing
from pydmdeep.types import Float2D

from pydmdeep.data import ToyDataSet


def run(data: dict[Any]):

    target_scaler = cast(
        MinMaxScaler|StandardScaler,
        data["prev_data"]["target_scaler"]
    )
    toydataset = cast(ToyDataSet, data["prev_data"]["dataset"])
    train_len = cast(float,data["prev_data"]["train_len"])
    xgrid = toydataset["xgrid"]
    tgrid = toydataset["tgrid"]

    X = toydataset["data"] # (nt,nx)
    train_idx = int(len(X)*train_len)
    X_train = X[:train_idx]
    X_train_scaled = target_scaler.transform(X_train)

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
    ax = ax.flatten()

    # plot original data
    im0 = ax[0].pcolor(xgrid, tgrid, X)
    ax[0].set_title("Original Data", fontsize=12)
    ax[0].set_xlabel("space", fontsize=12)
    ax[0].set_ylabel("Time Evolution", fontsize=12)

    # Apply extrapolation methods 
    d = 2
    dmd = DMD(svd_rank=d)
    dmd.fit(X_train_scaled.T)

    X_train_scaled_dmd = dmd.reconstructed_data.real.T


    X_test_scaled_dmd = []
    # x_pred = X_train_scaled[-1]
    x_pred = X_train_scaled_dmd[-1]
    for _ in X[train_idx:]:
        x_pred = dmd.predict(x_pred).real
        X_test_scaled_dmd.append(x_pred)
    X_test_scaled_dmd = np.array(X_test_scaled_dmd)

    X_train_dmd = target_scaler.inverse_transform(X_train_scaled_dmd)
    X_test_dmd = target_scaler.inverse_transform(X_test_scaled_dmd)
    X_dmd = np.vstack((X_train_dmd, X_test_dmd))

    split_idx = len(X_train_dmd)
    split_line = tgrid[split_idx][0]

    im1 = ax[1].pcolor(xgrid,tgrid,X_dmd)
    ax[1].set_title("DMD: No Hankel")
    ax[1].set_xlabel("space")
    ax[1].axhline(y=split_line,color='red',linestyle='--',label="Train-Test split")

    

    # Fit Classic w hankel dmd on train 
    d=2
    dmd = DMD(svd_rank=d*2)
    delay_dmd = hankel_preprocessing(dmd=dmd,d=d)

    delay_dmd.fit(X_train_scaled.T)

    X_train_scaled_delay_dmd = delay_dmd.reconstructed_data.real.T
    _, data_dim = X_train_scaled_delay_dmd.shape
    X_test_scaled_delay_dmd = []
    x_pred = X_train_scaled_delay_dmd[-d:].reshape(-1)
    for _ in X[train_idx:]:
        x_pred = delay_dmd.predict(x_pred).real
        X_test_scaled_delay_dmd.append(x_pred[data_dim:])
    X_test_scaled_delay_dmd = np.array(X_test_scaled_delay_dmd)

    X_train_delay_dmd = target_scaler.inverse_transform(X_train_scaled_delay_dmd)
    X_test_delay_dmd = target_scaler.inverse_transform(X_test_scaled_delay_dmd)

    X_delay_dmd = np.vstack((
        X_train_delay_dmd, X_test_delay_dmd
    ))

    im2 = ax[2].pcolor(xgrid,tgrid,X_delay_dmd)
    ax[2].set_title("DMD: Hankel")
    ax[2].set_xlabel("space")
    ax[2].axhline(y=split_line,color='red',linestyle='--',label="Train-Test split")

    # Lstm method
    X_lstm = data["time_delay_lstm_np"]
    im3 = ax[3].pcolor(xgrid[:-1], tgrid[:-1], X_lstm)
    ax[3].set_title("LSTM")
    ax[3].set_xlabel("space")
    ax[3].axhline(y=split_line,color='red',linestyle='--',label="Train-Test split")

    # DMD + LSTM method
    X_lstm_scaled = target_scaler.transform(X_lstm)
    d=2
    dmd = DMD(svd_rank=d*2)
    delay_dmd = hankel_preprocessing(dmd=dmd,d=d)
    delay_dmd.fit(X_lstm_scaled.T)
    X_dmd_lstm_scaled = delay_dmd.reconstructed_data.real.T
    X_dmd_lstm = target_scaler.inverse_transform(X_dmd_lstm_scaled)

    im4 = ax[4].pcolor(xgrid[:-1],tgrid[:-1], X_dmd_lstm)
    ax[4].set_title("DMD+LSTM")
    ax[4].set_xlabel("space")
    ax[4].axhline(y=split_line,color='red',linestyle='--',label="Train-Test split")

    X_test_true = X[train_idx:]
    no_hankel_err = L2_err(X_test_true,X_test_dmd)
    hankel_err = L2_err(X_test_true, X_test_delay_dmd)
    lstm_err = L2_err(X_test_true[:-1],X_lstm[train_idx:] )
    dmd_lstm_err = L2_err(X_test_true[:-1], X_dmd_lstm[train_idx:])

    errors = (no_hankel_err,hankel_err,lstm_err, dmd_lstm_err)

    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])
    fig.colorbar(im2, ax=ax[2])
    fig.colorbar(im3, ax=ax[3])
    fig.colorbar(im4, ax=ax[4])

    constructed_data = {
        "X": X,
        "X_dmd": X_dmd,
        "X_delay_dmd": X_delay_dmd,
        "X_lstm": X_lstm,
        "X_dmd_lstm": X_dmd_lstm,
    }

    return {"main": errors, "data": constructed_data}

def L2_err(X: Float2D, X_pred: Float2D) -> float: 
    return np.linalg.norm(X-X_pred) / np.linalg.norm(X)