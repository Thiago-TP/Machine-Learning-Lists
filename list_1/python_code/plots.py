import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt


def _multivariate_gaussian(x, avg, cov):
    return np.exp(-0.5 * (x - avg).T @ (np.linalg.inv(cov) @ (x - avg))) / (
        (2 * np.pi) * np.linalg.det(cov) ** 0.5
    )


def plot_multivariate_gaussian(
    avg: np.ndarray = np.array([-2, 1]),
    cov: np.ndarray = np.array([[1, -0.8], [-0.8, 4]]),
    output_folder: str = "./python_code/results/joint_pdf/",
) -> None:
    """Plots the multivariate gaussian distribution from question 3."""
    # Mean and standard deviation of each random variable
    m1, s1 = avg[0], np.sqrt(cov[0, 0])
    m2, s2 = avg[1], np.sqrt(cov[1, 1])

    # Samples: within "3 sigma" of averages
    x1 = np.linspace(m1 - 3 * s1, m1 + 3 * s1)
    x2 = np.linspace(m2 - 3 * s2, m2 + 3 * s2)

    # Plane x and y values of points that will be used
    X1, X2 = np.meshgrid(x1, x2)

    # Pointwise evaluation of the multivariate gaussian on the plane
    Z = np.array(
        [
            _multivariate_gaussian(np.array([x1, x2]), avg, cov)
            for x1, x2 in zip(X1.flatten(), X2.flatten())
        ]
    )
    # Reshape into 2 dimentions for compatibility with meshgrid outputs
    Z = np.reshape(Z, (x1.size, x2.size))

    # Plot the gaussian
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, linewidth=0)

    # Save different view of the plot
    for elev in [30, 90]:
        for azim in [60]:
            ax.view_init(elev=elev, azim=azim)
            if elev == 90:
                ax.set_zticklabels([])
            plt.savefig(
                output_folder + f"multivariate_gaussian_elev{elev}_azim{azim}.png",
                bbox_inches="tight",
                pad_inches=0,
            )
    plt.close()


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
        output_folder = f"./results/multivariate_linear_regression/{reference_realization.removesuffix(".mat")}/"
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
        output_folder = f"./results/multivariate_linear_regression/{reference_realization.removesuffix(".mat")}/"
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(
            output_folder + "regressions.pdf", bbox_inches="tight", pad_inches=0
        )
    plt.close()


if __name__ == "__main__":
    plot_multivariate_gaussian()
