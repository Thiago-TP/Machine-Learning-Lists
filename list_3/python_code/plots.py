"""This script plots Optuna optimization results and model structure for best models."""

from typing import Callable
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off oneDNN custom operations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # shush TensorFlow initialization messages

import optuna
import optuna.visualization.matplotlib as vis
import matplotlib.pyplot as plt
import keras
from keras.utils import plot_model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree


from classifiers.dt import DecisionTreeWrapper
from classifiers.fnn import FeedForwardNeuralNetworkWrapper
from classifiers.svm import SupportVectorMachineWrapper
from data import load_parkinson_detection, load_ppmi


def save_optuna_visualizations(
    wrapper, study: optuna.study.Study, output_dir: str = "plots"
) -> None:
    """
    Generate useful visualizations and save them as PDFs.
    """
    out = os.path.join(output_dir, wrapper.name)
    os.makedirs(out, exist_ok=True)

    plots: dict[str, Callable] = {
        "contour.pdf": vis.plot_contour,
        "edf.pdf": vis.plot_edf,
        "intermediate_values.pdf": vis.plot_intermediate_values,
        "optimization_history.pdf": vis.plot_optimization_history,
        "parallel_coordinate.pdf": vis.plot_parallel_coordinate,
        "param_importances.pdf": vis.plot_param_importances,
        "rank.pdf": vis.plot_rank,
        "slice.pdf": vis.plot_slice,
        "terminator_improvement.pdf": vis.plot_terminator_improvement,
    }

    for filename, func in plots.items():
        func(study)  # creates plot Axes object
        plt.savefig(os.path.join(out, filename), bbox_inches="tight", pad_inches=0)


def plot_best_fnn(wrapper, study: optuna.study.Study) -> None:
    plot_model(
        wrapper.load_best_model(study),
        to_file=f"plots/{wrapper.name}/best_model.pdf",
        show_shapes=True,
    )


def plot_best_dt(wrapper, study: optuna.study.Study) -> None:
    plot_tree(
        wrapper.load_best_model(study),
        feature_names=wrapper.context.feature_names,
        class_names=wrapper.context.class_names,
    )
    plt.savefig(
        f"plots/{wrapper.name}/best_model.pdf", bbox_inches="tight", pad_inches=0
    )


def plot_confusion_matrix(
    wrapper, study: optuna.study.Study, output_dir: str = "plots", cmap=plt.cm.Blues
) -> None:
    """
    Plot confusion matrix on test data using the best model from validation evaluation.
    An Optuna study must have been ran beforehand to set best hyperparameters.
    """

    test_preds = wrapper.load_best_model(study).predict(wrapper.context.x_test)
    if test_preds.ndim > 1:  # convert one-hot outputs (Keras standard) to integers
        test_preds = test_preds.argmax(axis=1)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay.from_predictions(
        wrapper.context.y_test, test_preds, cmap=cmap, colorbar=False
    )
    disp.ax_.set_title(wrapper.name.upper() + " Confusion Matrix")
    plt.savefig(
        f"{output_dir}/{wrapper.name}/confusion_matrix.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


if __name__ == "__main__":

    # Initialize data
    context_multiclass = load_ppmi()
    context_binaryclass = load_parkinson_detection()

    # Initialize wrappers
    fnn = FeedForwardNeuralNetworkWrapper(context_multiclass, "fnn")  # question 1
    dt = DecisionTreeWrapper(context_multiclass, "dt")  # question 2
    svm = SupportVectorMachineWrapper(context_binaryclass, "svm")  # question 3

    # Initiate or expand upon existing Optuna study
    for model in [dt, fnn, svm]:
        print(f"--- {model.name.upper()} Classifier ---")
        study = optuna.create_study(
            storage="sqlite:///optuna_study.db",  # database with all studies
            study_name=model.name,  # particular model study
            direction="minimize",
            load_if_exists=True,
        )

        print("Hyperparameters of best model:")
        print(study.best_params)

        print("Objective function value of best model:")
        print(study.best_value)

        plot_confusion_matrix(model, study)
        # save_optuna_visualizations(model, study)

        # if isinstance(model, FeedForwardNeuralNetworkWrapper):
        #     plot_best_fnn(model, study)
        # elif isinstance(model, DecisionTreeWrapper):
        #     plot_best_dt(model, study)
