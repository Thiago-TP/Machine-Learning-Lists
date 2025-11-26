# External libs -> model implementation
import optuna
import optuna.visualization.matplotlib as vis
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from sklearn.metrics import ConfusionMatrixDisplay

# Standard libraries -> type annotations and classes
from typing import Any
from pathlib import Path
from abc import ABC, abstractmethod

# Internal libraries -> input data handling
from data import DataContext


class GenericClassifier(ABC):
    def __init__(self, context: DataContext, name: str):
        self.context: DataContext = context
        self.name = name
        self.params: dict[str, Any] = None

    def suggest_hyperparams(self, trial: optuna.Trial) -> None:
        """Suggest hyperparameters for the current Optuna trial."""
        pass

    def build(self) -> Any:
        """
        Builds the model according to current hyperparameters.
        Keras/ScikitLearn Model comes out trained.
        """
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """Calculates model validation inaccuracy to be minimized by Optuna."""
        pass

    def run_optuna(self, n_trials=1, n_steps=10) -> None:
        """
        Run or resume an Optuna hyperparameter optimization study, saving results in HTML plots.
        Results are stored in an SQLite database file name "optuna_study.db".

        Parameters
        ---
        - n_trials: int
            Number of hyperparameter trials to run.
        - n_steps: int
            Number of intermediate evaluation steps per trial.

        Returns
        ---
            - None
        """
        study = optuna.create_study(
            storage="sqlite:///optuna_study.db",  # database with all studies
            study_name=self.name,  # particular model study
            direction="minimize",
            load_if_exists=True,
        )
        study.set_metric_names(["Validation Inaccuracy"])

        def objective(trial: optuna.Trial) -> float:
            # Run intermediate evaluations for trial pruning / comparing
            intermediate_values = []
            for step in range(n_steps):
                self.suggest_hyperparams(trial)
                intermediate_values.append(self.evaluate())
                trial.report(intermediate_values[-1], step=step)
                # Stop unpromising trials early
                if trial.should_prune():
                    raise optuna.TrialPruned()
            # Last evaluation taken as the objective value
            return intermediate_values[-1]

        study.optimize(objective, n_trials=n_trials)  # run optimization
        self.plot_confusion_matrix(study)  # save test results: confusion matrix
        self.save_optuna_visualizations(study)  # save Optuna-related visualizations

    def save_optuna_visualizations(
        self, study: optuna.study.Study, output_dir: str = "plots"
    ) -> None:
        """
        Generate useful visualizations and save them as HTML.
        """
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        out /= self.name
        out.mkdir(exist_ok=True)

        plots = {
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
            plt.savefig(out / filename, bbox_inches="tight", pad_inches=0)

        print(f"[Optuna] Saved visualizations to: {out.absolute()}")

    def plot_confusion_matrix(
        self, study: optuna.study.Study, output_dir: str = "plots", cmap=plt.cm.Blues
    ) -> None:
        """
        Plot confusion matrix on test data using the best model from validation evaluation.
        An Optuna study must have been ran beforehand to set best hyperparameters.
        """
        self.params.update(study.best_params)
        test_preds = self.build().predict(self.context.x_test)
        if test_preds.ndim > 1:  # convert one-hot outputs (Keras standard) to integers
            test_preds = test_preds.argmax(axis=1)

        disp = ConfusionMatrixDisplay.from_predictions(
            self.context.y_test, test_preds, cmap=cmap, colorbar=False
        )
        disp.ax_.set_title(self.name.upper() + " Confusion Matrix")
        plt.savefig(
            f"{output_dir}/{self.name}/confusion_matrix.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )


class KerasClassifier(GenericClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)
        # Standard hyperparameters for the FNN classifier
        self.params = {
            "batch_norm": True,  # apply batch normalization after layer
            "batch_size": 32,  # minibatch size
            "device": "/CPU:0",
            "dropout_rate": 0.0,  # dropout between layers
            "epochs": 15,  # number of epochs
            "h_activation": "relu",  # hidden layers' activation function
            "hidden_layer_width": 64,  # default width of a hidden layer
            "loss": "sparse_categorical_crossentropy",  # loss function
            "learn_rate": 0.001,  # base learning rate
            "learn_rate_schedule": "constant",  # "constant", "exp_decay", "cosine_decay"
            "metrics": ("accuracy",),  # metrics to consider
            "n_hidden_layers": 32,  # default number of hidden layers
            "optimizer": "adam",  # gradient descent optimizer
            "output_activation": "softmax",  # output activation function
        }

    @abstractmethod
    def suggest_hyperparams(self, trial: optuna.Trial) -> None:
        pass

    @abstractmethod
    def build(self) -> Model:
        pass

    def evaluate(self) -> float:

        val_preds = np.argmax(
            self.build().predict(
                self.context.x_val, batch_size=self.params["batch_size"], verbose=2
            ),
            axis=1,
        )
        val_accuracy = sum(self.context.y_val == val_preds) / len(val_preds)

        return 1.0 - val_accuracy


class SkLearnClassifier(GenericClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)

    @abstractmethod
    def suggest_hyperparams(self, trial: optuna.Trial):
        pass

    @abstractmethod
    def build(self) -> Any:
        pass

    def evaluate(self) -> float:
        model = self.build()
        preds = model.predict(self.context.x_val)
        val_accuracy = sum(self.context.y_val == preds) / len(self.context.y_val)
        return 1.0 - val_accuracy
