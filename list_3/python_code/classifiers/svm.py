import optuna
from sklearn.svm import SVC
from data import DataContext
from classifiers.generic_classifiers import SkLearnClassifier


class SupportVectorMachineWrapper(SkLearnClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)
        self.params = {
            "break_ties": False,
            "C": 1,
            "cache_size": 200,
            "class_weight": None,
            "coef0": 0.0,
            "decision_function_shape": "ovr",
            "degree": 3,
            "gamma": "scale",
            "kernel": "rbf",
            "max_iter": -1,
            "probability": False,
            "random_state": 242104677,
            "shrinking": True,
            "tol": 1e-3,
            "verbose": False,
        }

    def suggest_hyperparams(self, trial: optuna.Trial):
        self.params.update(
            {
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "poly", "rbf", "sigmoid"]
                ),
                "C": trial.suggest_float("C", 1e-1, 1e3, log=True),
            }
        )

    def build(self) -> SVC:
        return SVC(**self.params)
