from os import environ

environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off oneDNN custom operations
environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # shush TensorFlow initialization messages

from keras.models import load_model
from joblib import load

from classifiers.fnn import FeedForwardNeuralNetworkWrapper
from classifiers.dt import DecisionTreeWrapper
from data import prepare_data

if __name__ == "__main__":

    # Initialize data
    context = prepare_data()

    # Initialize wrappers
    fnn = FeedForwardNeuralNetworkWrapper(context, "fnn")
    dt = DecisionTreeWrapper(context, "dt")

    # Whether Optuna should run. If not, there should be a beskopen checkpoint model in "checkpoints"
    run_optuna = False

    # Run each model (train, validation, and test)
    for model in [dt, fnn]:
        print(f"--- Model {model.name.upper()} ---")
        if run_optuna:
            print(f"Running Optuna optimization for {model.name.upper()}...")
            model.run_optuna()

        # Keras' models use the module's load function
        try:
            _, model_acc = load_model(model.checkpoint_path).evaluate(
                context.X_test, context.y_test, verbose=0
            )
        # Scikit models use joblib's load
        except ValueError:
            test_preds = load(model.checkpoint_path).predict(context.X_test)
            model_acc = sum(test_preds == context.y_test) / len(test_preds)

        print(f"* Test accuracy: {100 * model_acc:.3f}%")
