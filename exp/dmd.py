# 
from pydmd import DMD
import matplotlib.pyplot as plt
from pydmdeep.data import ToyDataSet
from pydmd.preprocessing import hankel_preprocessing
from typing import cast

def run(data: dict):

    dataset = cast(ToyDataSet,data["prev_data"]["dataset"])
    X_orig = dataset["data"]
    x_grid = dataset["xgrid"]
    t_grid = dataset["tgrid"]

    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(13,5))
    ax = ax.flatten()

    # plot original data
    im0=ax[0].pcolor(x_grid,t_grid,X_orig)
    ax[0].set_title("Original Data", fontsize=15)
    ax[0].set_xlabel("space", fontsize=12)
    ax[0].set_ylabel("Time", fontsize=12)

    # plot Dmd Reconstruction w/o hankel 
    d=2
    dmd = DMD(svd_rank=4)
    delay_dmd = hankel_preprocessing(dmd,d=d)
    delay_dmd.fit(X_orig.T)

    X_dmd = delay_dmd.reconstructed_data.real.T
    im1=ax[1].pcolor(x_grid,t_grid,X_dmd)
    ax[1].set_title("DMD reconstruct",fontsize=15)
    ax[1].set_xlabel("space", fontsize=12)

    ax[2].set_title("DMD + LSTM", fontsize=15)
    ax[2].set_xlabel("space", fontsize=12)

    # plot DMD reconstruction w hankel

    # plot DMD+lstm
    fig.colorbar(im0,ax=ax[0])
    fig.colorbar(im1,ax=ax[1])

    # Do train/test split to see rollout 

    # Metric of different reconstruction task
    return {"main": None}