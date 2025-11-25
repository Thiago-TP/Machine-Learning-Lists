from optuna import load_study
from os import environ

environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off oneDNN custom operations
environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # shush TensorFlow initialization messages

from classifiers.dt import DecisionTreeWrapper
from classifiers.fnn import FeedForwardNeuralNetworkWrapper
from classifiers.svm import SupportVectorMachineWrapper
from data import load_parkinson_detection, load_ppmi


if __name__ == "__main__":

    # Initialize data
    context_multiclass = load_ppmi()
    context_binaryclass = load_parkinson_detection()

    # Initialize wrappers
    fnn = FeedForwardNeuralNetworkWrapper(context_multiclass, "fnn")  # question 1
    dt = DecisionTreeWrapper(context_multiclass, "dt")  # question 2
    svm = SupportVectorMachineWrapper(context_binaryclass, "svm")  # question 3

    # Whether Optuna should run.
    # If not, there should already be a study saved.
    # If a study already exists, running Optuna again will continue it.
    run_optuna = True

    # Run each model (train, validation, and test)
    for model, n_trials in zip([dt, svm, fnn], [25, 25, 25]):
        print(f"--- {model.name.upper()} Classifier ---")

        # Start or resume Optuna study
        if run_optuna:
            best_model = model.run_optuna(n_trials=n_trials)

        # Load best model from existing study
        else:
            study = load_study(
                storage="sqlite:///optuna_study.db",  # database with all studies
                study_name=model.name,  # particular model study
            )
            model.params.update(study.best_params)
            best_model = model.build()

        # Test the best model (predict is shared by Keras and Sklearn)
        test_preds = best_model.predict(model.context.x_test)
        if test_preds.ndim > 1:  # convert one-hot outputs to intergers
            test_preds = test_preds.argmax(axis=1)

        test_accuracy = sum(test_preds == model.context.y_test) / len(test_preds)
        print(f"[Test] Accuracy: {100 * test_accuracy:.2f}%\n")
