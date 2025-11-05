import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_best_weights(
    reference_realization: str,
    w: np.ndarray,
    features: np.ndarray,
    save_fig: bool = True,
):
    sns.clustermap(
        pd.DataFrame(dict(zip(features, w)), index=[0]),
        annot=True,
        figsize=(10, 3),
        # Disabling clustermap goodies since we're just interested in the heatmap
        row_cluster=False,
        col_cluster=False,
        cbar_pos=None,
        dendrogram_ratio=(0, 0),
        xticklabels=True,
        yticklabels=False,
    )
    plt.xticks(rotation=45, ha="right", fontsize=15)
    if save_fig:
        output_folder = f"./results/{reference_realization.removesuffix(".mat")}/"
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(
            output_folder + "best_weights.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()


def plot_regressions(
    reference_realization: str,
    best_realization: str,
    predictions: list[np.ndarray],
    best_fit: np.ndarray,
    r: np.ndarray,
    val_idx: int,
    test_idx: int,
    years: np.ndarray,
    target_feature: str,
    save_fig: bool = True,
):
    _, ax = plt.subplots(nrows=1, ncols=1)

    # Ensemble regression (except realization of best validation)
    for p in predictions:
        ax.plot(years, p, color="tab:gray", alpha=0.2)

    ax.plot(
        years,
        r,
        color="tab:blue",
        label=reference_realization.removesuffix(".mat"),
        alpha=0.8,
    )
    ax.plot(
        years,
        best_fit,
        color="tab:red",
        label=best_realization.removesuffix(".mat"),
        alpha=0.8,
    )

    # Vertical lines indicating where each period/datasets end
    ax.axvline(x=years[val_idx], color="tab:gray", linestyle="dashed")
    ax.axvline(x=years[test_idx], color="tab:gray", linestyle="dashed")

    ax.legend(loc="lower left")
    ax.grid(True)
    ax.set_title("Regression results")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(target_feature.replace("_", " ") + " (normalized units)")

    if save_fig:
        output_folder = f"./results/{reference_realization.removesuffix(".mat")}/"
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(
            output_folder + "regressions.pdf", bbox_inches="tight", pad_inches=0
        )
    plt.close()
