"""Runs multivariate linear regression algorithm to predict a given well's oil production rate"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt


def main(
    case_study: str,
    reference_realization: str,
    r_label: str,
    train_split: float,
    validation_split: float,
):
    # Set apart a realization for validation and testing

    reference_data, reference_outputs, years, features = load_data(
        case_study + reference_realization, r_label, is_case_study=True
    )

    # Pick time splits for training, validation and testing periods
    val_idx = int(train_split * len(years))
    test_idx = val_idx + int(validation_split * len(years))
    validation_set = reference_data[val_idx:test_idx, :]
    test_set = reference_data[test_idx:, :]

    # Train a model for each realization in the remaining ensemble
    ensemble = [
        case_study + mat
        for mat in os.listdir(case_study)
        if mat != reference_realization and mat.endswith(".mat")
    ]
    weight_matrices = [train(*load_data(e, r_label), val_idx) for e in ensemble]

    # From the trained models, pick the one with smallest validation error
    validation_errors = [
        validation(w, validation_set, reference_outputs[val_idx:test_idx])
        for w in weight_matrices
    ]
    best_w = weight_matrices[np.argmin(validation_errors)]

    print(validation_errors)
    print(best_w)
    print(features)

    # Test the "best" model
    predictions = test(best_w, reference_data)

    # Plot results
    plot_results(
        reference_realization,
        predictions,
        reference_outputs,
        val_idx,
        test_idx,
        years,
        r_label,
        unit="(m$^3$/s)",
    )

    return


def load_data(
    mat_file: str, r_label: str, is_case_study: bool = False
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):

    def _is_well_data(key: str) -> bool:
        """True if attribute refers to a well, False if not"""
        return "YEARS" in key or "OPR_PROD" in key or "WPR_PROD" in key

    # Loads data from selected realization
    data = pd.DataFrame(  # val is a single-value array so must be indexed
        {key: val[0] for key, val in loadmat(mat_file).items() if _is_well_data(key)}
    )

    r = data.pop(r_label).to_numpy()
    years = data.pop("YEARS").to_numpy()
    features = data.columns.to_numpy()
    # samples = (data - data.mean() / data.std()).to_numpy()
    samples = data.to_numpy()

    if is_case_study:
        return samples, r, years, features

    return samples, r


def train(samples: np.ndarray, r: np.ndarray, val_idx: int):
    # Limits data to training range, then adds a column of 1's at the left
    x = np.concat([np.ones((val_idx, 1)), samples[:val_idx, :]], axis=1)
    # Weight vector is then given by the normal equation
    return np.linalg.inv(x.T @ x) @ (x.T @ r[:val_idx])


def validation(w: np.ndarray, samples: np.ndarray, r: np.ndarray) -> float:
    # Adds a column of 1 at the left of samples
    x = np.concat([np.ones((r.size, 1)), samples], axis=1)
    # Returns RMSE
    error = r - (x @ w)
    return np.sqrt(np.dot(error, error) / len(error))


def test(w: np.ndarray, samples: np.ndarray):
    # Adds a column of 1 at the left of samples
    x = np.concat([np.ones((samples.shape[0], 1)), samples], axis=1)
    return x @ w


def plot_results(
    reference_realization: str,
    predictions: np.ndarray,
    r: np.ndarray,
    val_idx: int,
    test_idx: int,
    years: np.ndarray,
    target_feature: str,
    unit: str,
    r_train_color: str = "tab:gray",
    r_val_color: str = (1, 0.6, 0.6),
    r_test_color: str = "tab:red",
    train_color: str = "tab:gray",
    val_color: str = (0.6, 0.6, 1),
    test_color: str = "tab:blue",
):
    _, ax = plt.subplots(nrows=1, ncols=1)

    # Reference realization data r (training, validation and testing)
    ax.plot(years[:val_idx], r[:val_idx], color=r_train_color)
    ax.plot(years[val_idx:test_idx], r[val_idx:test_idx], color=r_val_color)
    ax.plot(years[test_idx:], r[test_idx:], color=r_test_color)

    # Regression results with from weights with best validation
    ax.plot(years[:val_idx], predictions[:val_idx], color=train_color, alpha=0.8)
    ax.plot(
        years[val_idx:test_idx],
        predictions[val_idx:test_idx],
        color=val_color,
        alpha=0.8,
    )
    ax.plot(years[test_idx:], predictions[test_idx:], color=test_color, alpha=0.8)

    # Vertical lines indicating where each period/datasets end
    ax.axvline(x=years[val_idx], color="black", linestyle="dashed")
    ax.axvline(x=years[test_idx], color="black", linestyle="dashed")

    # ax.legend()
    ax.grid(True)
    ax.set_title("Regression test results")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(target_feature.replace("_", " ") + " " + unit)

    save_file = "./results/" + reference_realization.removesuffix(".mat") + ".pdf"
    plt.savefig(save_file, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    args = {
        "case_study": "./data/EGG/",
        "reference_realization": "EGG11.mat",
        "r_label": "WOPR_PROD1",
        "train_split": 0.7,
        "validation_split": 0.15,
    }
    main(**args)
