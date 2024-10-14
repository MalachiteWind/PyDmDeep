import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_raw_frames(video, n_frames, n_rows, n_cols):
    # Calculate the number of frames to skip
    frameskip = len(video) / n_frames

    # Generate frame indices
    frame_ids = [int(frameskip * i) for i in range(n_frames)]

    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axs = axs.flatten()  # Flatten to easily index the axes

    for i, idx in enumerate(frame_ids):
        frame_t = video[idx]
        axs[i].imshow(frame_t)  # Display the frame
        axs[i].axis("off")  # Turn off axis
        axs[i].set_title(f"t = {idx}", fontsize=20)  # Set title with time point

    # Hide any remaining empty subplots if n_frames < n_rows * n_cols
    for j in range(i + 1, n_rows * n_cols):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()
    return


PICKLE_PATH = Path(__file__).parent.resolve() / "../../plume_videos/"


def _load_video(filename: str) -> tuple[np.ndarray, tuple[int, int]]:
    origin_filename = filename + "_ctr.pkl"
    with open(PICKLE_PATH / origin_filename, "rb") as fh:
        origin = pickle.load(fh)
    np_filename = filename[:-3] + "pkl"  # replace mov with pkl
    with open(PICKLE_PATH / np_filename, "rb") as fh:
        raw_vid = pickle.load(fh)
    orig_center = (int(origin[0]), int(origin[1]))
    return raw_vid, orig_center


def plot_lorenz_data(
    lorenz_data, n_rows: int = 3, n_cols: int = 1, figsize: tuple[int, int] = (5, 10)
):
    t = lorenz_data.t
    y = lorenz_data.y

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()

    titles = ["x(t)", "y(t)", "z(t)"]
    for i in range(len(y)):
        axs[i].plot(t, y[i])
        axs[i].set_ylabel(titles[i], fontsize=14)
        axs[i].set_xlabel("t", fontsize=14)

    plt.suptitle("Lorenz Solution")
    plt.show()
