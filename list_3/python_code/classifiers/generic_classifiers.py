# External libs -> model implementation
import optuna
import numpy as np
from keras import Model

# Standard libraries -> type annotations and classes
from typing import Any
from abc import ABC, abstractmethod

# Internal libraries -> input data handling
from data import DataContext


class GenericClassifier(ABC):
    def __init__(self, context: DataContext, name: str):
        self.context: DataContext = context
        self.name = name
        self.params: dict[str, Any] = None

    @abstractmethod
    def suggest_hyperparams(self, trial: optuna.Trial) -> None:
        """Suggest hyperparameters for the current Optuna trial."""
        pass

    @abstractmethod
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

    def load_best_model(self, study: optuna.study.Study) -> Any:
        """Loads the best model from a given Optuna study."""
        self.params.update(study.best_params)
        return self.build()

    def run_optuna(
        self, n_trials: int, n_steps: int = 10, database_name: str = "optuna_study.db"
    ) -> None:
        """
        Run or resume an Optuna hyperparameter optimization study.
        Results are stored in an SQLite database.

        Parameters
        ---
        - n_trials: int
            Number of hyperparameter trials to run.
            Must be 2 or greater if the study is brand new.
        - n_steps: int
            Number of intermediate evaluation steps per trial. Defaults to 10.
        - database_name: str
            Name of the .db file that will track trials and their results. Defaults to "optuna_study.db".

        Returns
        ---
        - None
        """
        study = optuna.create_study(
            storage=f"sqlite:///{database_name}",  # database with all studies
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


class KerasClassifier(GenericClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)

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
