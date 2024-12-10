# code from DMD tutorial
# https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial1/tutorial-1-dmd.ipynb
from typing import Literal
from typing import TypedDict

import numpy as np
from scipy.integrate import solve_ivp

from .types import Float1D
from .types import Float2D


class ToyDataSet(TypedDict):
    data: Float2D
    time_delay1: Float2D
    time_delay2: Float2D
    f1_data: Float2D
    f2_data: Float2D
    noisy_data: Float2D
    xgrid: Float2D
    tgrid: Float2D


class TimeDelayMatrices(TypedDict):
    time_delay1: Float2D
    time_delay2: Float2D


def generate_toy_dataset(
    w1: float = 2.3,
    w2: float = 2.8,
    tmin: float = 0,
    tmax: float = 4 * np.pi,
    xmin: float = -5,
    xmax: float = 5,
    nx: int = 65 * 4,
    nt: int = 129 * 4,
    noise_mean: float = 0,
    noise_std_dev: float = 0.2,
    seed: int = 1234,
) -> ToyDataSet:
    """
    Create 1-spatioal dimesion dataset consisting of two interacting spatial modes
    with differeing frequencies.

    Note: time_delays returned are transposed in convention with DMD notation.

    Parameters:
    ----------
    w1: frequency for first spatial mode.
    w2: frequency for second spatial mode.
    nx: number of spatial collocation points.
    nt: number of temporal collocation points.
    noise_mean: noise to add to data
    noise_std_dev: noise to add to data
    seed: random number generator seed

    Returns:
    --------
    X: data with shape (nt, nx)
    f1_data: first spatial mode corresponding to freq w1. Shape (nt, nx)
    f2_data: second spatial mode corresponding to freq w2. Shape (nt,nx)
    time_delay1: current state matrix for time delay. Shape (nx, nt-1)
    time_delay2: future state matrix for time delay. Shape (nx, nt-1)
    noisey_data: X with noise added
    x_grid: for plotting
    t_grid: for plotting

    """
    rng = np.random.default_rng(seed=seed)

    def f1(x, t):
        return 1.0 / np.cosh(x + 3) * np.cos(w1 * t)

    def f2(x, t):
        return 2.0 / np.cosh(x) * np.tanh(x) * np.sin(w2 * t)

    # Define the space and time grid for data collection.
    x = np.linspace(xmin, xmax, nx)
    t = np.linspace(tmin, tmax, nt)
    xgrid, tgrid = np.meshgrid(x, t)
    # dt = t[1] - t[0]  # time step between each snapshot

    # Data consists of 2 spatiotemporal signals.
    X1 = f1(xgrid, tgrid)
    X2 = f2(xgrid, tgrid)
    X = X1 + X2

    # Make a version of the data with noise.
    random_matrix = rng.normal(noise_mean, noise_std_dev, size=(nt, nx))
    Xn = X + random_matrix

    # Construct time delay matrices.
    time_delays = _construct_time_delay(X.T)
    Y1 = time_delays["time_delay1"]
    Y2 = time_delays["time_delay2"]

    # Construct noisy 
    time_delays_noise = _construct_time_delay(Xn.T)
    Y1_noise = time_delays_noise["time_delay1"]
    Y2_noise = time_delays_noise["time_delay2"]
    return ToyDataSet(
        data=X,
        time_delay1=Y1,
        time_delay2=Y2,
        f1_data=X1,
        f2_data=X2,
        noisy_data=Xn,
        xgrid=xgrid,
        tgrid=tgrid,
    )


def lorenz_ode(t: float, input: Literal[3], params: Literal[3]) -> Literal[3]:
    x, y, z = input
    sigma, rho, beta = params
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return np.array([x_dot, y_dot, z_dot])


def generate_lorenz_data(
    t_lin: Float1D, initial_cond: Float1D, ode_method: str = "LSODA"
) -> Float2D:
    sigma = 10
    rho = 28
    beta = 8 / 3

    def chaotic_lorenz(t, input):
        return lorenz_ode(t, input, params=(sigma, rho, beta))

    return solve_ivp(
        fun=chaotic_lorenz,
        t_span=[t_lin[0], t_lin[-1]],
        y0=initial_cond,
        t_eval=t_lin,
        method=ode_method,
    )


def _construct_time_delay(data: Float2D) -> TimeDelayMatrices:
    X = data[:, :-1]
    Y = data[:, 1:]
    return TimeDelayMatrices(time_delay1=X, time_delay2=Y)
