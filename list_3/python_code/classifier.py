# External libs -> model implementation
import optuna
import optuna.visualization as vis
import tensorflow as tf
from tensorflow.keras import models, callbacks, Model

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
    def train(self) -> float:
        pass

    def run_optuna(self, n_trials=30) -> None:

        study = optuna.create_study(
            direction="minimize",
            storage=f"sqlite:///{self.name}_optuna_study.db",
            study_name=f"{self.name}_optuna_study",
            load_if_exists=True,
        )

        def objective(trial):
            self.suggest_hyperparams(trial)
            return self.train()

        study.optimize(objective, n_trials=n_trials)
        print("\n\nBest hyperparameters so far:")
        print(study.best_trial.params)

        self.save_optuna_visualizations(study)

    def save_optuna_visualizations(
        self, study: optuna.study.Study, output_dir: str = "optuna_plots"
    ) -> None:
        """
        Generate useful visualizations and save them as HTML.
        """
        out = Path(output_dir) / self.name
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
            "batch_size": 32,  # minibatch size
            "device": "/CPU:0",
            "dropout_rate": 0.0,  # dropout between layers
            "epochs": 50,  # number of epochs
            "h_activation": "relu",  # hidden layers' activation function
            "hidden_layers": (64, 32),  # default architecture (width, amount)
            "loss": "sparse_categorical_crossentropy",  # loss function
            "lr": 0.001,  # base learning rate
            "lr_schedule": "constant",  # "constant", "exp_decay", "cosine_decay"
            "metrics": ("accuracy",),  # metrics to consider
            "monitor": "val_loss",  # early stopping control
            "optimizer_name": "adam",  # gradient descent optimizer
            "output_activation": "softmax",  # output activation function
            "patience": 10,  # epochs without improvement
            "restore_best_weights": True,
            "use_batch_norm": False,  # apply batch norm after Dense
        }

    @abstractmethod
    def suggest_hyperparams(self, trial: optuna.Trial) -> None:
        pass

    @abstractmethod
    def build(self) -> Model:
        pass

    def train(self) -> float:

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
            self.build().fit(
                self.context.X_train,
                self.context.y_train,
                validation_data=(self.context.X_val, self.context.y_val),
                epochs=self.params["epochs"],
                batch_size=self.params["batch_size"],
                callbacks=[early_stop, checkpoint],
                verbose=0,
            )

        best_model = models.load_model(self.checkpoint_path)
        val_loss = best_model.evaluate(
            self.context.X_val, self.context.y_val, verbose=0
        )[0]
        return val_loss


class SkLearnClassifier(GenericClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)

    @abstractmethod
    def suggest_hyperparams(self, trial: optuna.Trial):
        pass

    @abstractmethod
    def build(self) -> Any:
        pass

    def train(self):
        model = self.build()
        model.fit(self.context.X_train, self.context.y_train)
        preds = model.predict(self.context.X_val)
        val_loss = 1.0 - sum(self.context.y_val == preds)
        return val_loss
