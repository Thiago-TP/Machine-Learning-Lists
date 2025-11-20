import optuna
from sklearn.tree import DecisionTreeClassifier
from data import DataContext
from classifiers.generic_classifiers import SkLearnClassifier


class DecisionTreeWrapper(SkLearnClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)
        self.params = {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
        }

    def suggest_hyperparams(self, trial: optuna.Trial) -> None:
        self.params.update(
            {
                "criterion": trial.suggest_categorical(
                    "criterion", ["entropy", "gini", "log_loss"]
                ),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }
        )

    def build(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion=self.params["criterion"],
            max_depth=self.params["max_depth"],
            min_samples_split=self.params["min_samples_split"],
        )
