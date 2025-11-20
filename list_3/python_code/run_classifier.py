from keras import models
import joblib


from classifiers.fnn import FeedForwardNeuralNetworkWrapper
from classifiers.dt import DecisionTreeWrapper
from data import prepare_data

if __name__ == "__main__":

    # Initialize data
    context = prepare_data()

    # Initialize classifier wrappers
    fnn = FeedForwardNeuralNetworkWrapper(context, "fnn")
    dt = DecisionTreeWrapper(context, "dt")

    # Run Decision Tree
    print(f"Running Optuna optimization for {dt.name}...")
    dt.run_optuna()
    best_model = joblib.load(dt.checkpoint_path)
    test_preds = best_model.predict(context.X_test)
    model_acc = sum(test_preds == context.y_test) / len(test_preds)
    print(f"\nTest loss: Not applicable")
    print(f"Test accuracy: {100 * model_acc:.4f}%")

    # Run Feedforward Neural Network
    print(f"Running Optuna optimization for {fnn.name}...")
    fnn.run_optuna()
    model_loss, model_acc = models.load_model(fnn.checkpoint_path).evaluate(
        context.X_test, context.y_test, verbose=0
    )
    print(f"\nTest loss: {model_loss:.4f}")
    print(f"Test accuracy: {100 * model_acc:.4f}%")
