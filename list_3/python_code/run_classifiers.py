from os import environ

environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off oneDNN custom operations
environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # shush TensorFlow initialization messages

from keras.models import load_model
from joblib import load

from classifiers.dt import DecisionTreeWrapper
from classifiers.fnn import FeedForwardNeuralNetworkWrapper
from classifiers.svm import SupportVectorMachineWrapper
from data import DataContext, prepare_data


def load_pads() -> DataContext:
    return prepare_data(
        path="./data/multiclass_classification/",
        csv_name="ALAMEDA_PD_tremor_dataset.csv",
        label_columns=[  # last 4 columns are labels (source: https://zenodo.org/records/10782573)
            "Constancy_of_rest",
            "Kinetic_tremor",
            "Postural_tremor",
            "Rest_tremor",
        ],
        drop_columns=[
            "start_timestamp",  # irrelevant for classification
            "end_timestamp",  # irrelevant for classification
            "subject_id",  # irrelevant for classification
            "Magnitude_fft_dom_freq",  # always 0
            "Magnitude_fft_pw_ar_dom_freq",  # always 0
        ],
        plot_corr=False,
        summary=False,
    )


if __name__ == "__main__":

    # Initialize data
    context_multiclass = load_pads()
    context_binary_class = load_pads()  # replace with load_ppmi() when available

    # Initialize wrappers
    dt = DecisionTreeWrapper(context_multiclass, "dt")
    fnn = FeedForwardNeuralNetworkWrapper(context_multiclass, "fnn")
    svm = SupportVectorMachineWrapper(context_binary_class, "svm")

    # Whether Optuna should run. If not, there should be a beskopen checkpoint model in "checkpoints"
    run_optuna = True

    # Run each model (train, validation, and test)
    for model, n_trials in zip([dt, svm, fnn], [10, 10, 3]):
        print(f"--- {model.name.upper()} Classifier ---")
        if run_optuna:
            print(f"Running Optuna optimization for {model.name.upper()}...")
            model.run_optuna(n_trials=n_trials)

        # Keras' models use the module's load function
        try:
            _, model_acc = load_model(model.checkpoint_path).evaluate(
                model.context.x_test, model.context.y_test, verbose=0
            )
        # Scikit models use joblib's load
        except ValueError:
            test_preds = load(model.checkpoint_path).predict(model.context.x_test)
            model_acc = sum(test_preds == model.context.y_test) / len(test_preds)

        print(f"\t- Test accuracy: {100 * model_acc}%")
