import mitosis
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import Optional

trial_folder = Path(__file__).parents[2].absolute()/"trials"
image_folder = Path(__file__).parents[2].absolute()/"images"

trial_lookup = {
    5: "a56388", 
    10: "47e161",
    15: "055394",
    20: "1bd02a",
    25: "76dc56",
    30: "8144d2",
    35: "3237cb",
    40: "deb8b1",
    45: "14bc3c",
    50: "39a620",
    55: "3ecec2",
    60: "3d6e2b",
    65: "b32230",
    70: "9601eb",
    75: "63135b",
    80: "5a6d2e",
    85: "269c9d",
    90: "8781d5",
    95: "a4e0ae"
}


def run():
    plot_test_errors(save_path=image_folder/"test_errs.png")

def plot_test_errors(save_path:Optional[str]=None):
    test_errors = []
    train_ratios = range(5,100,5)
    for key in train_ratios:
        _, _, dmd_step = mitosis.load_trial_data(
            hexstr=trial_lookup[key],
            trials_folder=trial_folder/"dmd_results"
        )
        test_errors.append(dmd_step["main"])
    test_errors = np.array(test_errors)


    fig, ax = plt.subplots(1,1,figsize=(8,6),dpi=400)

    methods = ["DMD: no Hankel", "DMD: Hankel", "LSTM", "DMD+LSTM"]
    line_styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c']
    cut_off = 3

    for idx, method in enumerate(methods):
        ax.plot(train_ratios[cut_off:], test_errors[cut_off:, idx], label=method,
                linestyle=line_styles[idx], color=colors[idx], linewidth=2, markersize=8, marker='x')

    ax.legend(title="Methods", loc="upper right", fontsize=10, title_fontsize=12, frameon=False)
    ax.set_xlabel("Percentage of Observed Data", fontsize=12)
    ax.set_ylabel("Relative Test Error", fontsize=12)
    ax.set_title("Method Test Accuracy", fontsize=14, fontweight='bold')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Customize tick marks and labels
    ax.tick_params(axis='both', which='major', labelsize=10)

    # # Set axis limits (optional)
    # ax.set_xlim([train_ratios[0], train_ratios[-1]])
    ax.set_ylim([0 - 0.05, 1.5 + 0.05])

    # Add a horizontal line at y=0 for context (if appropriate)
    ax.axhline(0, color='black',linewidth=1, linestyle='--')

    if save_path:
        fig.savefig(save_path,bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()