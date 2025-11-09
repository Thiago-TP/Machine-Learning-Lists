import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.colors import ListedColormap

CMAP_CLASS = ListedColormap(["tab:blue", "tab:orange"])


def plot_multivariate_gaussian(
    avg: np.ndarray = np.array([-2, 1]),
    cov: np.ndarray = np.array([[1, -0.8], [-0.8, 4]]),
    output_folder: str = "./python_code/results/joint_pdf/",
) -> None:
    """Plots the multivariate gaussian distribution from question 3."""

    def _multivariate_gaussian(x, avg, cov):
        return np.exp(-0.5 * (x - avg).T @ (np.linalg.inv(cov) @ (x - avg))) / (
            (2 * np.pi) * np.linalg.det(cov) ** 0.5
        )

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


def plot_shuffling_effect(
    original_samples: pd.DataFrame,
    shuffled_samples: pd.DataFrame,
    save_as: str,
    proportion: float = 0.7,
) -> None:
    """Plots: original and shuffled datasets"""
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex=True, sharey=True)
    for ax, split, label in zip(
        axs.flatten()[:2],
        (original_samples, shuffled_samples),
        ("Original", "Shuffled"),
    ):
        # Points that would go to training set
        split.iloc[: int(proportion * split.shape[0]), :].plot.scatter(
            x="x1",
            y="x2",
            c="orange",
            title=f"{label} Set, {split.shape[0]} samples (100.00%)",
            ax=ax,
            grid=True,
            alpha=0.7,
            label="Training set",
        )
        # Points that would go to testing set
        split.iloc[int(proportion * split.shape[0]) :, :].plot.scatter(
            x="x1",
            y="x2",
            c="tab:red",
            ax=ax,
            grid=True,
            alpha=0.7,
            label="Testing set",
        )
        ax.set_xlabel("Feature $x_1$")
        ax.set_ylabel("Feature $x_2$")

    for ax, split in zip(axs.flatten()[2:], (original_samples, shuffled_samples)):
        # Points colored by initial index
        split.plot.scatter(
            x="x1",
            y="x2",
            colormap="Blues",
            c="original index",
            colorbar=False,
            ax=ax,
            grid=True,
            alpha=0.7,
        )
        ax.set_xlabel("Feature $x_1$")
        ax.set_ylabel("Feature $x_2$")

    # Colorbar adjustment
    cb = plt.colorbar(
        axs[1, 1].collections[0],
        ax=axs[1, 1],
        orientation="vertical",
        fraction=0.05,
        pad=0,
    )
    cb.set_label("Original index")

    # Save figure
    plt.savefig(save_as, bbox_inches="tight", pad_inches=0)


def plot_split(
    train_samples: pd.DataFrame, test_samples: pd.DataFrame, save_as: str
) -> None:
    """Plots training and testing datasets"""
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, split, title in zip(
        axs, (train_samples, test_samples), ("Training", "Testing")
    ):

        percentage = split.shape[0] / (train_samples.shape[0] + test_samples.shape[0])

        split.plot.scatter(
            x="x1",
            y="x2",
            c="class",
            colormap=CMAP_CLASS,
            colorbar=False,
            title=f"{title} Set, {split.shape[0]} samples ({percentage * 100 :.2f}%)",
            ax=ax,
            grid=True,
            alpha=0.7,
        )
        ax.set_xlabel("Feature $x_1$")
        ax.set_ylabel("Feature $x_2$")

    # Colorbar adjustment
    cb = plt.colorbar(
        axs[-1].collections[0],
        ticks=[-0.5, 0.5],
        ax=axs[1],
        orientation="vertical",
        fraction=0.1,
        pad=0,
    )
    cb.set_ticklabels([-1, 1])
    cb.set_label("Class")

    # Save figure
    plt.savefig(save_as, bbox_inches="tight", pad_inches=0)


def mahanalobis_ellipse(mean, cov, ax, n_std=3, **kwargs):
    # Eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Angle of the ellipse
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height of the ellipse
    width, height = 2 * n_std * np.sqrt(vals)

    # Create the ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)

    ax.add_patch(ellipse)
    return ellipse


def plot_estimates(
    train_samples: pd.DataFrame,
    mean_estimates: pd.DataFrame,
    covariance_estimates: pd.DataFrame,
    save_as: str,
    proportion: float = 0.7,
) -> None:
    _, axs = plt.subplots(figsize=(6, 6))
    train_samples.plot.scatter(
        x="x1",
        y="x2",
        c="class",
        colormap=CMAP_CLASS,
        colorbar=False,
        title=f"Training Set, {train_samples.shape[0]} samples ({proportion * 100 :.2f}%)",
        ax=axs,
        grid=True,
        alpha=0.7,
    )

    # Colorbar adjustment
    cb = plt.colorbar(
        axs.collections[0],
        ticks=[-0.5, 0.5],
        ax=axs,
        orientation="vertical",
        fraction=0.1,
        pad=0,
    )
    cb.set_ticklabels([-1, 1])
    cb.set_label("Class")

    plt.scatter(
        mean_estimates["x1"],
        mean_estimates["x2"],
        color="black",
        marker="+",
        s=100,
        label="Estimated Means",
    )

    for class_value in mean_estimates.index:
        mean = mean_estimates.loc[class_value].values
        cov = covariance_estimates.loc[class_value].values.reshape(2, 2)
        mahanalobis_ellipse(
            mean,
            cov,
            ax=axs,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            label=f"Estimated covariance" * (class_value == 1),  # only label once
        )

    axs.legend(loc="lower left")
    axs.set_xlabel("Feature $x_1$")
    axs.set_ylabel("Feature $x_2$")

    # Save figure
    plt.savefig(save_as, bbox_inches="tight", pad_inches=0)


def plot_test_errors(
    test_samples: pd.DataFrame, classifications: pd.DataFrame, save_as: str
) -> None:
    _, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=True, sharey=True)
    errors = classifications[
        classifications["class"] != test_samples["class"].reset_index(drop=True)
    ]

    for ax, dataset, title in zip(
        axs[:2],
        (test_samples, classifications),
        ("Expected class", "Assigned class"),
    ):
        dataset.plot.scatter(
            x="x1",
            y="x2",
            c="class",
            colormap=CMAP_CLASS,
            colorbar=False,
            ax=ax,
            grid=True,
            alpha=0.7,
        )
        # Red rectangle around the misslabeled samples
        ax.add_patch(
            Rectangle(
                (3, -5),
                3,
                5,
                facecolor="red",
                ec="red",
                alpha=0.2,
                linestyle="--",
                linewidth=2,
            )
        )
        ax.set_xlabel("Feature $x_1$")
        ax.set_ylabel("Feature $x_2$")
        ax.set_title(title)

    errors[errors["class"] == -1].plot.scatter(
        x="x1",
        y="x2",
        color="tab:blue",
        ax=axs[2],
        grid=True,
        alpha=0.7,
    )
    errors[errors["class"] == 1].plot.scatter(
        x="x1",
        y="x2",
        color="tab:orange",
        ax=axs[2],
        grid=True,
        alpha=0.7,
    )
    # Red rectangle around the misslabeled samples
    axs[2].add_patch(
        Rectangle(
            (3, -5),
            3,
            5,
            facecolor="red",
            ec="red",
            alpha=0.2,
            linestyle="--",
            linewidth=2,
        )
    )
    axs[2].set_xlabel("Feature $x_1$")
    axs[2].set_ylabel("Feature $x_2$")
    axs[2].set_title("Misclassifications")

    # Colorbar adjustment
    cb = plt.colorbar(
        axs[1].collections[0],
        ticks=[-0.5, 0.5],
        ax=axs[2],
        orientation="vertical",
        fraction=0.1,
        pad=0,
    )
    cb.set_ticklabels([-1, 1])
    cb.set_label("Class")

    # Save figure
    plt.savefig(save_as, bbox_inches="tight", pad_inches=0)


def plot_boundary(
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    classifications: pd.DataFrame,
    mean_estimates: pd.DataFrame,
    covariance_estimates: pd.DataFrame,
    boundary: np.array,
    X1: np.array,
    X2: np.array,
    save_as: str,
) -> None:
    _, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=True, sharey=True)
    for ax, classifications, title in zip(
        axs,
        (train_samples, test_samples, classifications),
        ("Train Set", "Test Set", "MAP Classifier"),
    ):
        classifications.plot.scatter(
            x="x1",
            y="x2",
            c="class",
            colormap=CMAP_CLASS,
            colorbar=False,
            title=title,
            ax=ax,
            grid=True,
            alpha=0.7,
        )

        ax.scatter(
            mean_estimates["x1"],
            mean_estimates["x2"],
            color="black",
            marker="+",
            s=100,
            label="Estimated Means",
        )

        for class_value in mean_estimates.index:
            mean = mean_estimates.loc[class_value].values
            cov = covariance_estimates.loc[class_value].values.reshape(2, 2)
            mahanalobis_ellipse(
                mean,
                cov,
                ax=ax,
                edgecolor="black",
                facecolor="none",
                linestyle="--",
                label=f"Estimated covariance" * (class_value == 1),  # only label once
            )

        # Red rectangle around the misslabeled samples
        ax.add_patch(
            Rectangle(
                (3, -5),
                3,
                5,
                facecolor="red",
                ec="red",
                alpha=0.2,
                linestyle="--",
                linewidth=2,
            )
        )

        # Decision boundary
        ax.contour(X1, X2, boundary, levels=[0], colors="black")
        ax.contourf(
            X1, X2, boundary, levels=[-1e9, 0], colors=["lightblue"], alpha=0.1
        )  # class -1 chosen
        ax.contourf(
            X1, X2, boundary, levels=[0, 1e9], colors=["gold"], alpha=0.1
        )  # class +1 chosen

        ax.legend(loc="lower left")
        ax.set_xlabel("Feature $x_1$")
        ax.set_ylabel("Feature $x_2$")

    # Colorbar adjustment
    cb = plt.colorbar(
        axs[0].collections[0],
        ticks=[-0.5, 0.5],
        ax=axs[:],
        orientation="horizontal",
        fraction=0.05,
        pad=0.125,
    )
    cb.set_ticklabels([-1, 1])
    cb.set_label("Class")

    # Save figure
    plt.savefig(save_as, bbox_inches="tight", pad_inches=0)


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
