"""Implements a MAP bivariate dichotomizer"""

import numpy as np
import pandas as pd

from plots import *


def train_test_split(
    original_dataset: str = "./python_code/data/bivariate_classifier/data.csv",
    p: float = 0.7,
    shuffle: bool = True,
    seed: int = 242104677,
) -> tuple[pd.DataFrame]:
    data = pd.read_csv(
        original_dataset,
        names=["x1", "x2", "class"],
    )
    data.index.name = "original index"
    data.reset_index(inplace=True)

    # Shuffle the dataset to avoid bias
    if shuffle:
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Calculate the split index
    split_index = int(len(data) * p)

    # Split the data
    train_set = data.iloc[:split_index, :]
    test_set = data.iloc[split_index:, :]

    return data, train_set, test_set


def log_odds_discriminant(x, mean1, cov1, prior1, mean2, cov2, prior2) -> float:
    """Evaluates the discriminant of the classifier for the given sample and parameters."""
    inv_cov1 = np.linalg.inv(cov1)
    inv_cov2 = np.linalg.inv(cov2)
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    q_x1 = -0.5 * (x - mean1).T @ inv_cov1 @ (x - mean1)
    q_x2 = -0.5 * (x - mean2).T @ inv_cov2 @ (x - mean2)
    quadratic_term = q_x1 - q_x2
    linear_term = -0.5 * (np.log(det_cov1) - np.log(det_cov2))
    constant_term = np.log(prior1 / prior2)

    return quadratic_term + linear_term + constant_term


def test(
    test_set: pd.DataFrame,
    prior_estimates: pd.DataFrame,
    mean_estimates: pd.DataFrame,
    covariance_estimates: pd.DataFrame,
) -> pd.DataFrame:
    classifications = {"x1": [], "x2": [], "class": []}

    mean1 = mean_estimates.loc[1].values
    cov1 = covariance_estimates.loc[1].values.reshape(2, 2)
    prior1 = prior_estimates.loc[1]

    mean2 = mean_estimates.loc[-1].values
    cov2 = covariance_estimates.loc[-1].values.reshape(2, 2)
    prior2 = prior_estimates.loc[-1]

    for _, row in test_set.iterrows():

        x = row[["x1", "x2"]].values
        g = log_odds_discriminant(x, mean1, cov1, prior1, mean2, cov2, prior2)
        assigned_class = 1 if g > 0 else -1

        classifications["x1"].append(row["x1"])
        classifications["x2"].append(row["x2"])
        classifications["class"].append(assigned_class)

    return pd.DataFrame(classifications)


def print_performance(classifications: pd.DataFrame) -> pd.DataFrame:
    errors = classifications[
        classifications["class"] != test_samples["class"].reset_index(drop=True)
    ]

    fp = len(errors[errors["class"] == 1])
    fn = errors.shape[0] - fp
    tp = (test_samples["class"] == 1).sum() - fp
    tn = (test_samples["class"] == -1).sum() - fn

    # Performance metrics
    return pd.DataFrame(
        {
            "precision": 100 * (tp + tn) / (tp + tn + fp + fn),
            "accuracy": 100 * tp / (tp + fp),
            "recall": 100 * tp / (tp + fn),
        },
        index=["Value (%)"],
    ).T


def calculate_boundary(
    prior_estimates: pd.DataFrame,
    mean_estimates: pd.DataFrame,
    covariance_estimates: pd.DataFrame,
    npoints: int = 200,
    minx1: int = -6,
    maxx1: int = 10,
    minx2: int = -12,
    maxx2: int = 12,
) -> tuple[np.array, np.array, np.array]:
    # Create grid
    x1 = np.linspace(minx1, maxx1, npoints)
    x2 = np.linspace(minx2, maxx2, npoints)
    X1, X2 = np.meshgrid(x1, x2)

    # Extract parameters for calculations in the discriminant
    mean1 = mean_estimates.loc[1].values
    cov1 = covariance_estimates.loc[1].values.reshape(2, 2)
    prior1 = prior_estimates.loc[1]
    mean2 = mean_estimates.loc[-1].values
    cov2 = covariance_estimates.loc[-1].values.reshape(2, 2)
    prior2 = prior_estimates.loc[-1]

    # Calculates the discriminant pointwise on the plane
    B = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            B[i, j] = log_odds_discriminant(
                [X1[i, j], X2[i, j]], mean1, cov1, prior1, mean2, cov2, prior2
            )
    return B, X1, X2


def save_results(
    original_samples: pd.DataFrame,
    shuffled_samples: pd.DataFrame,
    prior_estimates: pd.DataFrame,
    mean_estimates: pd.DataFrame,
    covariance_estimates: pd.DataFrame,
    performance: pd.DataFrame,
    boundary: np.array,
    X1: np.array,
    X2: np.array,
    root: str = "./python_code/results/bivariate_classifier/",
) -> None:
    # Estimates
    with open(root + "estimates.txt", "w") as f:
        print(f"Prior estimates:\n{prior_estimates}\n", file=f)
        print(f"Mean estimates:\n{mean_estimates}\n", file=f)
        print(f"Covariance estimates:\n{covariance_estimates}\n", file=f)

    # Precision, accuracy, recall
    with open(root + "performance.txt", "w") as f:
        print(performance, file=f)

    # Effects of shuffling
    plot_shuffling_effect(
        original_samples, shuffled_samples, root + "shuffling_effect.png"
    )
    plot_split(train_samples, test_samples, root + "split.png")
    plot_estimates(
        train_samples,
        mean_estimates,
        covariance_estimates,
        root + "parameters_estimates.png",
    )
    plot_test_errors(test_samples, classifications, root + "test_errors.png")
    plot_boundary(
        train_samples,
        test_samples,
        classifications,
        mean_estimates,
        covariance_estimates,
        boundary,
        X1,
        X2,
        root + "boundary.png",
    )


if __name__ == "__main__":
    # Apply the split
    shuffled_samples, train_samples, test_samples = train_test_split()

    # Estimates likelihoods parameters
    prior_estimates = train_samples["class"].value_counts(normalize=True).sort_index()
    mean_estimates = train_samples.groupby("class")[["x1", "x2"]].mean()
    covariance_estimates = train_samples.groupby("class")[["x1", "x2"]].cov()

    # Runs test samples through the discriminant
    classifications = test(
        test_samples, prior_estimates, mean_estimates, covariance_estimates
    )

    # Check out precision, accuracy, and recall of results
    performance = print_performance(classifications)

    # Calculate boundary (roots of the discriminant)
    boundary, X1, X2 = calculate_boundary(
        prior_estimates, mean_estimates, covariance_estimates
    )

    # Save results
    original_samples, _, _ = train_test_split(shuffle=False)
    save_results(
        original_samples,
        shuffled_samples,
        prior_estimates,
        mean_estimates,
        covariance_estimates,
        performance,
        boundary,
        X1,
        X2,
    )
