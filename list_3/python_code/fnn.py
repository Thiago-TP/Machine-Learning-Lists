from optuna import Trial
from tensorflow.keras import layers, models, optimizers, Input, Model
from classifier import KerasClassifier

from data import DataContext


class FNNClassifier(KerasClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)

    def build_learning_rate_schedule(self):
        if self.params["lr_schedule"] == "constant":
            return self.params["lr"]

        elif self.params["lr_schedule"] == "exp_decay":
            return optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.params["lr"],
                decay_steps=100,
                decay_rate=0.96,
                staircase=True,
            )

        elif self.params["lr_schedule"] == "cosine_decay":
            return optimizers.schedules.CosineDecay(
                initial_learning_rate=self.params["lr"],
                decay_steps=self.params["epochs"] * 50,
            )

        raise ValueError(f"Unknown lr_schedule: {self.params["lr_schedule"]}")

    def suggest_hyperparams(self, trial: Trial) -> None:
        # Parameters Optuna will vary to find best case scenario
        self.params.update(
            {
                "hidden_layers": [
                    trial.suggest_int("h1", 16, 256),
                    trial.suggest_int("h2", 16, 256),
                ],
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "batch_norm": trial.suggest_categorical("bn", [False, True]),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            }
        )

    def build(self) -> Model:
        model = models.Sequential()

        # First hidden layer (with input shape)
        model.add(
            layers.Dense(
                Input(shape=self.params["hidden_layers"]),
                activation=self.params["h_activation"],
            )
        )

        if self.params["use_batch_norm"]:
            model.add(layers.BatchNormalization())

        if self.params["dropout_rate"] > 0:
            model.add(layers.Dropout(self.params["dropout_rate"]))

        # Additional layers
        for width in self.params["hidden_layers"][1:]:
            model.add(layers.Dense(width, activation=self.params["h_activation"]))

            if self.params["use_batch_norm"]:
                model.add(layers.BatchNormalization())

            if self.params["dropout_rate"] > 0:
                model.add(layers.Dropout(self.params["dropout_rate"]))

        # Output layer
        model.add(
            layers.Dense(
                self.context.num_classes, activation=self.params["output_activation"]
            )
        )

        # Optimizer with LR schedule
        lr_schedule = self.build_learning_rate_schedule()
        if self.params["optimizer_name"] == "adam":
            optimizer = optimizers.Adam(learning_rate=lr_schedule)
        else:
            optimizer = optimizers.SGD(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer,
            loss=self.params["loss"],
            metrics=self.params["metrics"],
        )

        return model
