"""Runs multivariate linear regression algorithm to predict a given well's oil production rate"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

from plots import *


def main(
    case_study: str,
    reference_realization: str,
    r_label: str,
    train_split: float,
    validation_split: float,
):
    # Set apart a realization for validation and testing
    reference_data, r, years, features = load_data(
        case_study + reference_realization, r_label, is_case_study=True
    )

    # Pick time splits for training, validation and testing periods
    val_idx = int(train_split * len(years))
    test_idx = val_idx + int(validation_split * len(years))

    # Train a model for each realization in the remaining ensemble
    realizations = [
        mat
        for mat in os.listdir(case_study)
        if mat != reference_realization and mat.endswith(".mat")
    ]
    ensemble = [case_study + mat for mat in realizations]
    weight_matrices = [train(load_data(e, r_label), r, val_idx) for e in ensemble]

    # From the trained models, pick the one with smallest validation error
    predictions = [reference_data @ w for w in weight_matrices]
    errors = [r - p for p in predictions]
    validation_errors = [np.linalg.norm(e[val_idx:test_idx]) for e in errors]

    # Indicate best realization
    index_of_best = np.argmin(validation_errors)
    best_validation = predictions.pop(index_of_best)
    best_weights = weight_matrices.pop(index_of_best)
    name_of_best = realizations[index_of_best]

    # Plots
    plot_best_weights(reference_realization, best_weights, features)

    plot_regressions(
        reference_realization,
        name_of_best,
        predictions,
        best_validation,
        r,
        val_idx,
        test_idx,
        years,
        r_label,
    )

    return


def load_data(
    mat_file: str, r_label: str, is_case_study: bool = False, normalize: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    def _is_well_data(key: str) -> bool:
        """True if attribute refers to a well, False if not"""
        return "YEARS" in key or "OPR_PROD" in key or "WPR_PROD" in key

    # Loads data from selected realization
    data = pd.DataFrame(  # val is a single-value array so must be indexed
        {key: val[0] for key, val in loadmat(mat_file).items() if _is_well_data(key)}
    )

    years = data.pop("YEARS").to_numpy()
    data = (data - data.mean()) / data.std()  # normalization of features
    r = data.pop(r_label).to_numpy()
    features = data.columns.to_numpy()
    samples = data.to_numpy()

    if is_case_study:
        return samples, r, years, features

    return samples


def train(samples: np.ndarray, reference: np.ndarray, val_idx: int):
    # Restricts data to training period
    x = samples[:val_idx, :]
    r = reference[:val_idx]
    # Weight vector is then given by the normal equation
    return np.linalg.inv(x.T @ x) @ (x.T @ r)


if __name__ == "__main__":
    args = {
        "case_study": "./python_code/data/multivariate_linear_regression/EGG/",
        "r_label": "WOPR_PROD1",
        "train_split": 0.5,
        "validation_split": 0.25,
    }
    for ref in os.listdir(args["case_study"]):
        main(reference_realization=ref, **args)
