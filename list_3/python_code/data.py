import numpy as np
from dataclasses import dataclass


@dataclass
class DataContext:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    num_features: int
    num_classes: int


def prepare_data(
    train_split: float = 0.7, validation_split: float = 0.15
) -> DataContext:
    # Synthetic data
    num_samples = 1000
    num_features = 4
    num_classes = 3

    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples).astype(np.int32)

    train_size = int(train_split * num_samples)
    val_size = int(validation_split * num_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]

    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    return DataContext(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        num_features=num_features,
        num_classes=num_classes,
    )
