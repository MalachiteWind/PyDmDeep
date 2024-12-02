from pathlib import Path
import mitosis
import matplotlib.pyplot as plt
from typing import Optional
from exp.post.test_errors import trial_lookup

trials_folder = Path(__file__).parents[2].absolute() / "trials"
image_folder = Path(__file__).parents[2].absolute() / "images"

def run():
    hexstr = trial_lookup[45]
    plot_original_data(
        hexstr=hexstr,
        save_path=image_folder/"orig.png"
    )
def plot_original_data(hexstr: str, save_path: Optional[str] = None):
    data_step, _, _= mitosis.load_trial_data(
        hexstr=hexstr, 
        trials_folder=trials_folder/"dmd_results"
    )

    xgrid = data_step["data"]["dataset"]["xgrid"]
    tgrid = data_step["data"]["dataset"]["tgrid"]
    X = data_step["data"]["dataset"]["data"]

    fig, ax = plt.subplots(1,1,figsize=(15,6),dpi=400)
    ax.pcolor(tgrid,xgrid,X)
    ax.set_ylabel("Features", fontsize=18, fontweight='bold')
    ax.set_xlabel("Time", fontsize=18, fontweight='bold')
    ax.set_title(r"$\mathbf{f(x,t) = f_1(x,t) + f_2(x,t)}$", fontsize=26)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')


if __name__ == "__main__":
    run()