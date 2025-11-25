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
    label_column: str,
    drop_columns: Iterable[str] | None = None,
    skiprows: int = 0,
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
    # Load data from CSV assingning integers to categorical features
    data = (
        pd.read_csv(path + csv_name, skiprows=skiprows)
        .drop(columns=drop_columns)
        .dropna()
        .apply(lambda x: pd.factorize(x)[0] if x.dtype == "object" else x)
        .reset_index(drop=True)
    )

    # Parse raw data into samples and labels
    x, y = data.drop(columns=[label_column]), data[[label_column]]

    # Remove highly correlated features from samples
    hcp = highly_correlated_pairs(x, threshold=correlation_threshold)
    x_f = x.drop(columns=hcp["x2"].unique())

    # Plot correlations before and after filtering
    # May take a while to finish for large datasets
    if plot_corr:
        plot_correlations(x, path + "original_correlations.pdf", figsize=(20, 20))
        plot_correlations(x_f, path + "filtered_correlations.pdf", figsize=(20, 20))

    # Log effects of correlation filtering
    num_classes = np.max(y) + 1
    num_samples, num_features = x_f.shape

    # Shuffles then parses non highly correlated data into DataContext
    rand_inds = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(rand_inds)

    x_shuffled = x_f.to_numpy(dtype=np.float64)[rand_inds]
    y_shuffled = y.to_numpy(dtype=np.int64)[rand_inds].ravel()

    train_size = int(train_split * num_samples)
    val_size = int(validation_split * num_samples)

    context = DataContext(
        x_train=x_shuffled[:train_size],
        y_train=y_shuffled[:train_size],
        x_val=x_shuffled[train_size : train_size + val_size],
        y_val=y_shuffled[train_size : train_size + val_size],
        x_test=x_shuffled[train_size + val_size :],
        y_test=y_shuffled[train_size + val_size :],
        num_features=num_features,
        num_classes=num_classes,
    )

    if summary:
        write_filtering_info(x, context, hcp)
        write_label_info(context)

    return context


def write_filtering_info(
    x: pd.DataFrame, context: DataContext, hcp: pd.DataFrame
) -> None:

    print("=== CORRELATION FILTERING SUMMARY ===")
    print(f"- Samples: {x.shape[0]}")
    print(f"- Features: {context.num_features}")
    print(f"- Classes: {context.num_classes}")
    print(f"- Original features: {x.shape[1]}")
    print(f"- Removed features: {x.shape[1] - context.num_features}")
    print(f"- Reduction: {100 * (1 - context.num_features / x.shape[1]):.2f}%")
    print(f"- Removed features:")
    if not hcp.empty:
        for feature in sorted(hcp["x2"].unique()):
            print(f"\t- {feature}")
    else:
        print("\t- None")
    print("=====================================\n")


def write_label_info(context: DataContext):
    print("=== LABEL SUMMARY ===")
    for mode, y in {
        "Train": context.y_train,
        "Validation": context.y_val,
        "Test": context.y_test,
    }.items():
        print(f"{mode} Labels:")
        for l in np.unique(y):
            occurrences = sum(y == l)
            print(
                f"\t- '{l}': {occurrences} samples ({100 * occurrences / y.shape[0]:.2f}%)"
            )
    print("=====================\n")


# --- APIs for loading assignment datasets ---


def load_parkinson_detection(
    plot_corr: bool = False, summary: bool = False
) -> DataContext:
    # Oxford Parkinson's Disease Detection Dataset
    # https://archive.ics.uci.edu/dataset/174/parkinsons
    return prepare_data(
        path="./data/binary_classification/",
        csv_name="parkinsons.csv",
        drop_columns=["name"],
        label_column="status",  # 1 for PD, 0 for healthy
        plot_corr=plot_corr,
        summary=summary,
    )


def load_ppmi(plot_corr: bool = False, summary: bool = False) -> DataContext:
    # PPMI multiclass classification dataset
    # https://www.ppmi-info.org/access-data-specimens/download-data
    return prepare_data(
        path="./data/multiclass_classification/",
        csv_name="meta_data.11192021.csv",
        drop_columns=[  # irrelevant/redundant for classification at hand
            "HudAlphaSampleName",
            "Small RNA-Seq",
            "Long RNA-seq",
            "PATNO",
            "PATNO Visit",
            "PoolAssign",
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
        label_column="Case Control",  # strictly multiclass ("Case", "Control", "Other")
        plot_corr=plot_corr,
        summary=summary,
    )


# --------------------------------------------


if __name__ == "__main__":
    # Test loading functions
    _ = load_parkinson_detection(plot_corr=True, summary=True)
    _ = load_ppmi(plot_corr=True, summary=True)
