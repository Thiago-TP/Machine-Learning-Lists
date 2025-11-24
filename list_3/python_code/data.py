from typing import Any, Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class DataContext:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    num_features: int
    num_classes: int


def highly_correlated_pairs(
    df: pd.DataFrame, threshold: float = 0.9, numeric_types: list[Any] = [np.number]
) -> pd.DataFrame:
    """
    Returns a long-form DataFrame of feature pairs whose absolute correlation is greater than a given threshold.
    Columns: ["x1", "x2", "corr"] where "corr" is the signed Pearson correlation.
    Only numeric columns are considered and each pair appears once (xi, xj with i < j).
    """
    numeric = df.select_dtypes(include=numeric_types)
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["x1", "x2", "corr"])

    # Pipeline:
    # 1. compute Pearson correlation matrix
    # 2. mask to get upper triangle (lower triangle + diagonal become NaN)
    # 3. stack to long form (multi-index Series with index: xi, [xi+1, xi+2...]; values: corr)
    # 4. flatten multi-index to DataFrame (columns: xi, xj, corr)
    # 5. remove pairs below threshold (i.e., the valid pairs)
    corr = numeric.corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().reset_index()
    pairs.columns = ["x1", "x2", "corr"]
    return pairs[pairs["corr"].abs() >= threshold]


def plot_correlations(
    samples: pd.DataFrame,
    save_path: str,
    figsize: tuple[int, int] = (20, 20),
    hmap_kwargs: dict[str, Any] = {
        "annot": True,
        "cmap": "coolwarm",
        "center": 0,
        "square": True,
        "fmt": ".2f",
    },
):
    """Plot and save a heatmap of the Pearson correlations between features in samples."""
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(samples.corr(), ax=ax, **hmap_kwargs)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.clf()


def prepare_data(
    path: str,
    csv_name: str,
    label_columns: Iterable[str],
    drop_columns: Iterable[str] | None = None,
    correlation_threshold: float = 0.9,
    train_split: float = 0.7,
    validation_split: float = 0.15,
    seed: int = 242104677,
    plot_corr: bool = False,
    summary: bool = False,
) -> DataContext:
    """
    Prepare data for classification tasks by loading a CSV into a DataContext object. Features are removed if highly correlated.

    Parameters:
    ---
        - path (str): Directory path to the CSV file (must include / at the end).
        - csv_name (str): Name of the CSV file including .csv suffix.
        - label_columns (Iterable[str]): Columns to use as labels.
        - drop_columns (Iterable[str] | None): Columns to drop from the dataset before processing. Defaults to None
        - correlation_threshold (float): Threshold above which features are considered, between 0 and 1. Defaults to 0.9.
        - train_split (float): Proportion of data to use for training, between 0 and 1. Defaults to 0.7.
        - validation_split (float): Proportion of data to use for validation, between 0 and 1. Defaults to 0.15.
        - plot_corr (bool): Whether to plot and save correlation heatmaps before and after filtering. Defaults to True.
        - seed (int): Random seed for shuffling data. Defaults to 242104677.
        - summary (bool): Whether to print a summary of the correlation filtering process. Defaults to True.
    """
    # Parse raw data into samples and labels
    data = pd.read_csv(path + csv_name).drop(columns=drop_columns).dropna()
    data = data.apply(lambda x: pd.factorize(x)[0] if x.dtype == "object" else x)
    data.reset_index(drop=True, inplace=True)
    x, y = data.drop(columns=label_columns), data[label_columns]

    # Remove highly correlated features from samples
    hcp = highly_correlated_pairs(x, threshold=correlation_threshold)
    x_f = x.drop(columns=hcp["x2"].unique())

    # Plot correlations before and after filtering
    # May take a while to finish for large datasets
    if plot_corr:
        plot_correlations(x, path + "original_correlations.png", figsize=(60, 60))
        plot_correlations(x_f, path + "filtered_correlations.png", figsize=(20, 20))

    # Log effects of correlation filtering
    num_classes = np.max(y) + 1 if y.shape[1] == 1 else y.shape[1]
    num_samples, num_features = x_f.shape
    if summary:
        print("=== CORRELATION FILTERING SUMMARY ===")
        print(f"- Samples: {num_samples}")
        print(f"- Features: {num_features}")
        print(f"- Classes: {num_classes}")
        print(f"- Original features: {x.shape[1]}")
        print(f"- Removed features: {x.shape[1] - num_features}")
        print(f"- Reduction: {100 * (1 - num_features / x.shape[1]):.2f}%")
        print(f"- Removed features:")
        for feature in sorted(hcp["x2"].unique()):
            print(f"\t- {feature}")
        print("=====================================\n")

    # Shuffles then parses non highly correlated data into DataContext
    rand_inds = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(rand_inds)

    x_shuffled = x_f.to_numpy(dtype=np.float64)[rand_inds]
    y_shuffled = y.to_numpy(dtype=np.int64)[rand_inds]
    if y_shuffled.shape[1] == 1:
        y_shuffled = y_shuffled.ravel()

    train_size = int(train_split * num_samples)
    val_size = int(validation_split * num_samples)

    return DataContext(
        x_train=x_shuffled[:train_size],
        y_train=y_shuffled[:train_size],
        x_val=x_shuffled[train_size : train_size + val_size],
        y_val=y_shuffled[train_size : train_size + val_size],
        x_test=x_shuffled[train_size + val_size :],
        y_test=y_shuffled[train_size + val_size :],
        num_features=num_features,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    # Use examples / test on assignment's datasets

    # Alameda PADS multilabel classification dataset
    # https://zenodo.org/records/10782573
    multilabel_context = prepare_data(
        path="./data/multiclass_classification/",
        csv_name="ALAMEDA_PD_tremor_dataset.csv",
        label_columns=[  # last 4 columns are labels
            "Constancy_of_rest",
            "Kinetic_tremor",
            "Postural_tremor",
            "Rest_tremor",
        ],
        drop_columns=[
            "start_timestamp",  # irrelevant for classification at hand
            "end_timestamp",  # irrelevant for classification
            "subject_id",  # irrelevant for classification
            "Magnitude_fft_dom_freq",  # always 0
            "Magnitude_fft_pw_ar_dom_freq",  # always 0
        ],
        summary=True,
    )
    print(multilabel_context.y_train)

    # PPMI multiclass classification dataset
    # https://www.ppmi-info.org/access-data-specimens/download-data
    multiclass_context = prepare_data(
        path="./data/binary_classification/",
        csv_name="meta_data.11192021.csv",
        drop_columns=[  # irrelevant/redundant for classification at hand
            "HudAlphaSampleName",
            "Small RNA-Seq",
            "Long RNA-seq",
            "PATNO",
            "PATNO Visit",
            "Phase",
            "Clinical Event",
            "Month",
            "Age (Bin)",
            "Age at diagnosis",
            "Box",
            "Position",
            "Plate",
            "Neutrophil Score",
            "Basophils (%)",
            "Eosinophils (%)",
            "Lymphocytes (%)",
            "Neutrophils (%)",
            "Neutrophil/Lymphocyte",
            "RBC Morphology",
            "Usable Bases (%)",
            "Multimapped (%)",
            "Uniquely mapped (%)",
            "Total reads",
            "UPDRS1 score",
            "UPDRS2 score",
            "UPDRS3 score",
            "UPDRS4 score",
            "UPDRS totscore",
            "UPSIT",
            "moca",
            # Redundant with "Case Control" label
            "Disease Status",  # strictly multiclass ("PD", "SWEDD", "Healthy Control", etc)
            "Study Arm",  # strictly multiclass ("PD", "SWEDD", "Healthy Control", etc)
            "Diagnosis",  # strictly multiclass ("PD", "SWEDD", "Healthy Control", etc)
        ],
        label_columns=[
            "Case Control",  # strictly multiclass ("Case", "Control", "Other")
        ],
        summary=True,
    )
    print(multiclass_context.y_train)
