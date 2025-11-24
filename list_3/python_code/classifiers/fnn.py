import optuna
import tensorflow as tf
from keras import layers, models, optimizers, Model
from classifiers.generic_classifiers import KerasClassifier

from data import DataContext


class FeedForwardNeuralNetworkWrapper(KerasClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)

    def build_learning_rate_schedule(self):
        if self.params["learn_rate_schedule"] == "constant":
            return self.params["learn_rate"]

        elif self.params["learn_rate_schedule"] == "exp_decay":
            return optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.params["learn_rate"],
                decay_steps=100,
                decay_rate=0.96,
                staircase=True,
            )

        elif self.params["learn_rate_schedule"] == "cosine_decay":
            return optimizers.schedules.CosineDecay(
                initial_learning_rate=self.params["learn_rate"],
                decay_steps=self.params["epochs"] * 50,
            )

        raise ValueError(
            f"Unknown learn_rate_schedule: {self.params["learn_rate_schedule"]}"
        )

    def suggest_hyperparams(self, trial: optuna.Trial) -> None:
        self.params.update(
            {
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
                "h_activation": trial.suggest_categorical(
                    "h_activation", ["relu", "sigmoid"]
                ),
                "hidden_layer_width": trial.suggest_int("hidden_layer_width", 1, 8),
                "learn_rate": trial.suggest_float("learn_rate", 1e-4, 1e-2, log=True),
                "n_hidden_layers": trial.suggest_int("n_hidden_layers", 1, 8),
            }
        )

    def build(self) -> Model:
        model = models.Sequential()

        # Input layer
        model.add(layers.InputLayer((self.context.num_features,)))
        if self.params["batch_norm"]:
            model.add(layers.BatchNormalization())

        # Hidden layers
        for _ in range(self.params["n_hidden_layers"]):
            model.add(
                layers.Dense(
                    self.params["hidden_layer_width"],
                    activation=self.params["h_activation"],
                )
            )
            if self.params["batch_norm"]:
                model.add(layers.BatchNormalization())
            if self.params["dropout_rate"] > 0:
                model.add(layers.Dropout(self.params["dropout_rate"]))

        # Output layer
        model.add(
            layers.Dense(
                self.context.num_classes,
                activation=self.params["output_activation"],
            )
        )

        # Compiling with learning rate schedule
        lr_schedule = self.build_learning_rate_schedule()
        if self.params["optimizer"] == "adam":
            optimizer = optimizers.Adam(learning_rate=lr_schedule)
        else:
            optimizer = optimizers.SGD(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer,
            loss=self.params["loss"],
            metrics=list(self.params["metrics"]),
        )

        with tf.device(self.params["device"]):
            model.fit(
                self.context.x_train,
                self.context.y_train,
                epochs=self.params["epochs"],
                batch_size=self.params["batch_size"],
                shuffle=False,  # data is already shuffled
                verbose=0,
            )

        return model
