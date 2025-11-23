# External libs -> model implementation
import optuna
import optuna.visualization as vis
import tensorflow as tf
from keras import callbacks, Model
import joblib

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
        self.checkpoint_path: Path = None

    def suggest_hyperparams(self, trial: optuna.Trial) -> None:
        pass

    def build(self) -> Any:
        pass

    @abstractmethod
    def evaluate(self) -> float:
        pass

    def run_optuna(self, n_trials=2) -> None:

        study = optuna.create_study(
            storage=f"sqlite:///optuna_study.db",  # database with all studies
            study_name=self.name,  # particular model study
            direction="minimize",
            load_if_exists=True,
        )

        def objective(trial) -> float:
            self.suggest_hyperparams(trial)
            return self.evaluate()

        study.optimize(objective, n_trials=n_trials)
        print("\n\nBest hyperparameters so far:")
        for t in study.best_trials:
            print(t.params)

        self.save_optuna_visualizations(study)

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
        # History of the last trained model
        self.history = None
        # Keras filepath to last trained model
        self.checkpoint_path = Path("checkpoints") / Path(
            f"best_optuna_{self.name}.keras"
        )
        # Standard hyperparameters for the FNN classifier
        self.params = {
            "batch_norm": False,  # apply batch norm after Dense
            "batch_size": 32,  # minibatch size
            "device": "/CPU:0",
            "dropout_rate": 0.0,  # dropout between layers
            "epochs": 50,  # number of epochs
            "h_activation": "relu",  # hidden layers' activation function
            "hidden_layer_width": 64,  # default width of a hidden layer
            "loss": "sparse_categorical_crossentropy",  # loss function
            "learn_rate": 0.001,  # base learning rate
            "learn_rate_schedule": "constant",  # "constant", "exp_decay", "cosine_decay"
            "metrics": ("accuracy",),  # metrics to consider
            "monitor": "accuracy",  # early stopping control
            "n_hidden_layers": 32,  # default number of hidden layers
            "optimizer": "adam",  # gradient descent optimizer
            "output_activation": "softmax",  # output activation function
            "patience": 10,  # epochs without improvement
            "restore_best_weights": True,
        }

    @abstractmethod
    def suggest_hyperparams(self, trial: optuna.Trial) -> None:
        pass

    @abstractmethod
    def build(self) -> Model:
        pass

    def evaluate(self) -> float:

        early_stop = callbacks.EarlyStopping(
            monitor=self.params["monitor"],
            patience=self.params["patience"],
            restore_best_weights=self.params["restore_best_weights"],
        )

        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(self.checkpoint_path),
            save_best_only=True,
            monitor=self.params["monitor"],
        )

        with tf.device(self.params["device"]):
            model = self.build()
            model.fit(
                self.context.x_train,
                self.context.y_train,
                validation_data=(self.context.x_val, self.context.y_val),
                epochs=self.params["epochs"],
                batch_size=self.params["batch_size"],
                callbacks=[early_stop, checkpoint],
                verbose=0,
            )
            model.save(self.checkpoint_path, include_optimizer=False)

        _, val_accuracy = model.evaluate(
            self.context.x_val, self.context.y_val, verbose=0
        )

        return 1.0 - val_accuracy


class SkLearnClassifier(GenericClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)
        self.checkpoint_path = Path("checkpoints")
        self.checkpoint_path.mkdir(exist_ok=True)
        self.checkpoint_path /= Path(f"best_optuna_{self.name}.pkl")

    @abstractmethod
    def suggest_hyperparams(self, trial: optuna.Trial):
        pass

    @abstractmethod
    def build(self) -> Any:
        pass

    def evaluate(self) -> float:
        model = self.build()
        model.fit(self.context.x_train, self.context.y_train)
        joblib.dump(model, self.checkpoint_path)

        preds = model.predict(self.context.x_val)
        val_accuracy = sum(self.context.y_val == preds) / len(self.context.y_val)
        return 1.0 - val_accuracy
