from typing import Optional

import matplotlib.pyplot as plt
from plumex.config import data_lookup

from ..data import generate_toy_dataset
from ..data import ToyDataSet
from .utils import _load_video
from .utils import plot_raw_frames


def plot_toy_dataset(dataset: Optional[ToyDataSet] = None):

    if dataset:
        pass
    else:
        dataset = generate_toy_dataset()

    titles = ["$f_1(x,t)$", "$f_2(x,t)$", "$f = f_1 + f_2$"]
    data = [dataset["f1_data"], dataset["f2_data"], dataset["data"]]

    _ = plt.figure(figsize=(15, 5), dpi=600)
    for n, title, d in zip(range(131, 134), titles, data):
        plt.subplot(n)
        plt.pcolor(dataset["xgrid"], dataset["tgrid"], d.real)
        plt.title(title, fontsize=20, fontweight="bold")
        plt.xlabel("Space", fontsize=18)
        plt.ylabel("Time", fontsize=18)
        plt.colorbar()
    plt.show()


def plot_plumes_image():
    video, orig_center_fc = _load_video(data_lookup["filename"]["low-869"])
    plot_raw_frames(video=video[599:], n_frames=4, n_rows=1, n_cols=4)

def run():
    plot_toy_dataset()
    plot_plumes_image()

if __name__ == "__main__":
    run()
