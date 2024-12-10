from pathlib import Path
import mitosis
import matplotlib.pyplot as plt
from typing import Optional
from exp.post.test_errors import trial_lookup

trials_folder = Path(__file__).parents[2].absolute() / "trials"
image_folder = Path(__file__).parents[2].absolute() / "images"

def run():
    train_percent = 30
    hexstr_45=trial_lookup[train_percent]
    save_path = image_folder / "methods.png"
    # save_path = None
    # plot_methods(hexstr_45, save_path=save_path)
    plot_methods(hexstr_45,save_path=image_folder/"methods2.png", rotate=True)



def plot_methods(hexstr:str, save_path: Optional[str]=None, rotate:bool=False):
    data_step, _, dmd_step = mitosis.load_trial_data(
    hexstr=hexstr,trials_folder=trials_folder/"dmd_results"
)

    train_len = data_step["data"]["train_len"]
    xgrid = data_step["data"]["dataset"]["xgrid"]
    tgrid = data_step["data"]["dataset"]["tgrid"]

    X = dmd_step["data"]["X"]
    X_dmd = dmd_step["data"]["X_dmd"]
    X_delay_dmd = dmd_step["data"]["X_delay_dmd"]
    X_lstm = dmd_step["data"]["X_lstm"]
    X_dmd_lstm = dmd_step["data"]["X_dmd_lstm"]

    titles = ["Original: f(x,t)", "DMD: No Hankel", "DMD: Hankel", "LSTM", "DMD+LSTM"]

    if rotate:
        fig, ax = plt.subplots(nrows=1,ncols=5,dpi=400,figsize=(18,10))
    else:
        fig, ax = plt.subplots(nrows=5,ncols=1, dpi=400, figsize=(12,18))

    if rotate:
        arg1, arg2 = xgrid, tgrid
    else: 
        arg1, arg2 = tgrid, xgrid

    ax[0].pcolor(arg1, arg2, X)
    ax[1].pcolor(arg1, arg2, X_dmd)
    ax[2].pcolor(arg1, arg2, X_delay_dmd)
    ax[3].pcolor(arg1[:-1], arg2[:-1], X_lstm)
    ax[4].pcolor(arg1[:-1],arg2[:-1],X_dmd_lstm)


    split_idx = int(len(tgrid)*train_len)
    split_line = tgrid[split_idx][0]
    for i in range(5):
        if rotate:
            ax[i].axhline(y=split_line,color='red',linestyle='--',linewidth=4)
            ax[i].set_xlabel("Features", fontsize=14, fontweight='bold')
        else:
            ax[i].axvline(x=split_line,color='red',linestyle='--',linewidth=4)
            ax[i].set_ylabel("Space",fontsize=14, fontweight='bold')

        ax[i].set_title(titles[i],fontsize=18,fontweight='bold')

    if rotate:
        ax[0].set_ylabel("Time Evolution", fontsize=14,fontweight='bold')
    else:
        ax[4].set_xlabel("Time Evolution",fontsize=14, fontweight='bold')
    if save_path:
        fig.savefig(save_path,bbox_inches='tight')


if __name__ == "__main__":
    run()



