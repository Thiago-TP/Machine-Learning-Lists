from os import environ

environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off oneDNN custom operations
environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # shush TensorFlow initialization messages

import optuna
from keras.utils import plot_model

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
            "PoolAssign",
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
    # context_binaryclass = todo!()

    # Initialize wrappers
    dt = DecisionTreeWrapper(context_multiclass, "dt")
    fnn = FeedForwardNeuralNetworkWrapper(context_multiclass, "fnn")
    # svm = SupportVectorMachineWrapper(context_binaryclass, "svm")

    # Whether Optuna should run.
    # If not, there should already be a study saved.
    # If a study already exists, running Optuna again will continue it.
    run_optuna = False

    # Run each model (train, validation, and test)
    for model in [dt, fnn]:
        print(f"--- {model.name.upper()} Classifier ---")

        # Start or resume Optuna study
        if run_optuna:
            best_model = model.run_optuna()

        # Load best model from existing study
        else:
            study = optuna.load_study(
                storage="sqlite:///optuna_study.db",  # database with all studies
                study_name=model.name,  # particular model study
            )
            model.params.update(study.best_params)
            best_model = model.build()

        # Test the best model (predict is shared by Keras and Sklearn)
        test_preds = best_model.predict(model.context.x_test)
        if test_preds.ndim > 1:  # convert one-hot outputs to intergers
            test_preds = test_preds.argmax(axis=1)
            plot_model(best_model, to_file=f"{model.name}.png", show_shapes=True)

        test_accuracy = sum(test_preds == model.context.y_test) / len(test_preds)
        print(f"[Test] Accuracy: {100 * test_accuracy:.2f}%\n")
