from tensorflow.keras import models
from fnn import FNNClassifier

# from dt import DecisionTreeClassifier
# from svm import SVMClassifier
from data import prepare_data


if __name__ == "__main__":

    context = prepare_data()
    fnn = FNNClassifier(context, "fnn")
    # dt = DecisionTreeClassifier(context, "dt")
    # svm = SVMClassifier(context, "svm")

    # Find best hyperparameters on validation set using Optuna's framework
    print("Running Optuna optimization...")
    fnn.run_optuna()

    # Evaluate best model on test set
    model = models.load_model(fnn.checkpoint_path)
    test_loss, test_acc = model.evaluate(context.X_test, context.y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
