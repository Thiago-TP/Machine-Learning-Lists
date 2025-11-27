"""This script runs an Optuna hyperparameter optimization study for multiple classifiers.
Each classifier wrapper implements its own hyperparameter suggestions and evaluation method.
Results are saved in an SQLite database from which visualizations/confusion matrices can be made later.
Classifiers included:
    - Feed-Forward Neural Network (FNN) for multiclass classification (PPMI dataset)
    - Decision Tree (DT) for multiclass classification (PPMI dataset)
    - Support Vector Machine (SVM) for binary classification (Parkinson's Detection dataset)
"""

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

    # Initiate or expand upon existing Optuna study
    for model in [dt, svm, fnn]:
        print(f"--- {model.name.upper()} Classifier ---")
        model.run_optuna(n_trials=100)
