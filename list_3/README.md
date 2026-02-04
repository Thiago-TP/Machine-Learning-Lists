# List 3 Overview

The structure of this list is as follows:

- `python_code/`: Contains Python implementations of machine learning classifiers and data processing.
    - `run_classifiers.py`: Main script for executing classifier training and evaluation.
    - `data.py`: Data loading and preprocessing utilities.
    - `plots.py`: Visualization utilities for generating analysis plots.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `data_log.txt`: Log file for tracking data processing execution.
    - `classifiers/`: Directory containing classifier implementations.
        - `generic_classifiers.py`: Base classifier interface and utilities.
        - `dt.py`: Decision Tree classifier implementation.
        - `svm.py`: Support Vector Machine classifier implementation.
        - `fnn.py`: Feedforward Neural Network classifier implementation.
    - `data/`: Directory containing input datasets for each classification task.
        - `binary_classification/`: Binary classification datasets.
            - `parkinsons.csv`: Parkinson's disease classification data.
        - `multiclass_classification/`: Multiclass classification datasets.
            - `meta_data.11192021.csv`: Multiclass classification data.
    - `plots/`: Directory storing output visualizations organized by classifier.
        - `dt/`: Decision Tree analysis plots.
        - `svm/`: Support Vector Machine analysis plots.
        - `fnn/`: Feedforward Neural Network analysis plots.

- `references/`: Directory containing reference materials and external resources.

- `report/`: LaTeX project containing the complete analysis report.
    - `report.tex`: Main report document.
    - `config.tex`: Report configuration and styling.
    - `references.bib`: Bibliography and citation references.
    - `figures/`: Directory with LaTeX figure definitions.
        - `cover.tex`: Report cover page.
        - `flowchart-preprocessing.tex`: Data preprocessing flowchart.
    - `sections/`: Directory containing report sections.
        - `note_on_data.tex`: Data documentation and notes.
        - `optuna_results.tex`: Hyperparameter optimization results.
        - `question_1.tex` through `question_3.tex`: Analysis sections.
            
> [!NOTE]
> The `report` directory and `questions.pdf` have already been addressed in this project's [main README.md](./../README.md).