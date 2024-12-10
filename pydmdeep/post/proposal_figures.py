from typing import Optional
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from plumex.config import data_lookup

from pydmdeep.data import generate_toy_dataset
from pydmdeep.data import ToyDataSet
from pydmdeep.post.utils import _load_video
from pydmdeep.post.utils import plot_raw_frames

image_folder = Path(__file__).parents[2].absolute() / "images"

def plot_toy_dataset(dataset: Optional[ToyDataSet] = None, save_path: Optional[str] = None):

    if dataset:
        pass
    else:
        dataset = generate_toy_dataset()

    titles = ["$\mathbf{f_1(x,t)}$", "$\mathbf{f_2(x,t)}$", "$\mathbf{f = f_1 + f_2}$"]
    data = [dataset["f1_data"], dataset["f2_data"], dataset["data"]]

    fig = plt.figure(figsize=(15, 8), dpi=600)
    for n, title, d in zip(range(131, 134), titles, data):
        plt.subplot(n)
        plt.pcolor(dataset["xgrid"], dataset["tgrid"], d.real)
        plt.title(title, fontsize=20, fontweight="bold")
        plt.xlabel("Features", fontsize=18, fontweight='bold')
        plt.ylabel("Time", fontsize=18, fontweight='bold')
        plt.colorbar()
    plt.show()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')



def plot_plumes_image():
    video, orig_center_fc = _load_video(data_lookup["filename"]["low-869"])
    plot_raw_frames(video=video[599:], n_frames=4, n_rows=1, n_cols=4)


def run():
    save_path = image_folder / "dataset_f1_f2_f.png"
    dataset = generate_toy_dataset(tmax=8 * np.pi, nt=129 * 8, nx=65 * 8)
    plot_toy_dataset(dataset=dataset, save_path=save_path)
    # plot_plumes_image()


if __name__ == "__main__":
    run()
