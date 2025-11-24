from os import environ

environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off oneDNN custom operations
environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # shush TensorFlow initialization messages

from keras.models import load_model
from joblib import load

from classifiers.dt import DecisionTreeWrapper
from classifiers.fnn import FeedForwardNeuralNetworkWrapper
from classifiers.svm import SupportVectorMachineWrapper
from data import DataContext, prepare_data


def load_ppmi() -> DataContext:

    # PPMI multiclass classification dataset
    # https://www.ppmi-info.org/access-data-specimens/download-data
    return prepare_data(
        path="./data/binary_classification/",
        csv_name="meta_data.11192021.csv",
        drop_columns=[  # irrelevant/redundant for classification at hand
            "HudAlphaSampleName",
            "Small RNA-Seq",
            "Long RNA-seq",
            "PATNO",
            "PATNO Visit",
            "Phase",
            "Clinical Event",
            "Month",
            "Age (Bin)",
            "Age at diagnosis",
            "Box",
            "Position",
            "Plate",
            "Neutrophil Score",
            "Basophils (%)",
            "Eosinophils (%)",
            "Lymphocytes (%)",
            "Neutrophils (%)",
            "Neutrophil/Lymphocyte",
            "RBC Morphology",
            "Usable Bases (%)",
            "Multimapped (%)",
            "Uniquely mapped (%)",
            "Total reads",
            "UPDRS1 score",
            "UPDRS2 score",
            "UPDRS3 score",
            "UPDRS4 score",
            "UPDRS totscore",
            "UPSIT",
            "moca",
            # Redundant with "Case Control" label
            "Disease Status",  # strictly multiclass ("PD", "SWEDD", "Healthy Control", etc)
            "Study Arm",  # strictly multiclass ("PD", "SWEDD", "Healthy Control", etc)
            "Diagnosis",  # strictly multiclass ("PD", "SWEDD", "Healthy Control", etc)
        ],
        label_columns=[
            "Case Control",  # strictly multiclass ("Case", "Control", "Other")
        ],
    )


if __name__ == "__main__":

    # Initialize data
    context_multiclass = load_ppmi()

    # Initialize wrappers
    dt = DecisionTreeWrapper(context_multiclass, "dt")
    fnn = FeedForwardNeuralNetworkWrapper(context_multiclass, "fnn")
    svm = SupportVectorMachineWrapper(context_multiclass, "svm")

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
