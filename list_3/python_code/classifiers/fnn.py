import optuna
import tensorflow as tf
from keras import layers, models, optimizers, Model
from classifiers.generic_classifiers import KerasClassifier

from data import DataContext


class FeedForwardNeuralNetworkWrapper(KerasClassifier):
    def __init__(self, context: DataContext, name: str):
        super().__init__(context, name)
        self.params = {
            "batch_size": 32,  # minibatch size
            "device": "/CPU:0",  # whether to use CPu or GPU
            "epochs": 15,  # number of epochs
            "h_activation": "relu",  # hidden layers' activation function
            "hidden_layer_width": 64,  # default width of a hidden layer
            "loss": "sparse_categorical_crossentropy",  # loss function
            "n_hidden_layers": 32,  # default number of hidden layers
            "optimizer": "adam",  # gradient descent optimizer
            "output_activation": "softmax",  # output activation function
        }

    def suggest_hyperparams(self, trial: optuna.Trial) -> None:
        self.params.update(
            {
                "h_activation": trial.suggest_categorical(
                    "h_activation", ["relu", "sigmoid"]
                ),
                "hidden_layer_width": trial.suggest_int("hidden_layer_width", 1, 16),
                "learn_rate": trial.suggest_float("learn_rate", 1e-4, 1e-2, log=True),
                "n_hidden_layers": trial.suggest_int("n_hidden_layers", 1, 16),
            }
        )

    def build(self) -> Model:
        model = models.Sequential()

        # Input layer
        model.add(layers.Dense(units=self.context.num_features, activation=None))

        # Hidden layers
        for _ in range(self.params["n_hidden_layers"]):
            model.add(
                layers.Dense(
                    units=self.params["hidden_layer_width"],
                    activation=self.params["h_activation"],
                )
            )

        # Output layer
        model.add(
            layers.Dense(
                self.context.num_classes,
                activation=self.params["output_activation"],
            )
        )

        # Compiling with optimizer
        optimizer = (
            optimizers.Adam()
            if self.params["optimizer"] == "adam"
            else optimizers.SGD()
        )

        model.compile(optimizer=optimizer, loss=self.params["loss"])

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
