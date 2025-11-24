# External libs -> model implementation
import optuna
import optuna.visualization as vis
import numpy as np
from keras import Model

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
        pass

    def build(self) -> Any:
        pass

    @abstractmethod
    def evaluate(self) -> float:
        pass

    def run_optuna(self, n_trials=5, n_steps=10) -> Any:

        study = optuna.create_study(
            storage="sqlite:///optuna_study.db",  # database with all studies
            study_name=self.name,  # particular model study
            direction="minimize",
            load_if_exists=True,
        )

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
        self.save_optuna_visualizations(study)  # save visualizations
        self.params.update(study.best_params)  # update params with best found
        return self.build()  # returns ready-to-test model with best params

    def save_optuna_visualizations(
        self, study: optuna.study.Study, output_dir: str = "optuna_plots"
    ) -> None:
        """
        Generate useful visualizations and save them as HTML.
        """
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        out /= self.name
        out.mkdir(exist_ok=True)

        plots = {
            "optimization_history.html": vis.plot_optimization_history(study),
            "param_importances.html": vis.plot_param_importances(study),
            "parallel_coordinate.html": vis.plot_parallel_coordinate(study),
            "contour.html": vis.plot_contour(study),
            "slice.html": vis.plot_slice(study),
        }

        for filename, fig in plots.items():
            fig.write_html(out / filename)

        print(f"[Optuna] Saved visualizations to: {out.absolute()}")


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
