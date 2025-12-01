import optuna
from sklearn.svm import SVC
from data import DataContext
from classifiers.generic_classifiers import SkLearnClassifier


class SupportVectorMachineWrapper(SkLearnClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)
        self.params = {"C": 1, "coef0": 0.0, "degree": 3, "kernel": "rbf"}

    def suggest_hyperparams(self, trial: optuna.Trial):
        args = {
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "C": trial.suggest_float("C", 1e-1, 1e3, log=True),
            "degree": (
                trial.suggest_int("degree", 2, 5)
                if self.params["kernel"] == "poly"
                else 3
            ),
            "coef0": (
                trial.suggest_float("coef0", 0.0, 10.0)
                if self.params["kernel"] in ["poly", "sigmoid"]
                else 0.0
            ),
        }
        self.params.update(args)

    def build(self) -> SVC:
        return SVC(**self.params).fit(self.context.x_train, self.context.y_train)
