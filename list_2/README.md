# List 2 Overview 

The structure of this list is as follows:

- `rust_code/`: Contains Rust implementations of machine learning algorithms.
    - `src/`: Source code directory with algorithm implementations.
        - `main.rs`: Main entry point for the Rust project.
        - `question_1.rs`: Implementation for K-Means clustering.
        - `question_2.rs`: Implementation for Logistic Regression.
        - `question_3.rs`: Implementation for Principal Component Analysis.
        - `handle_csv.rs`: Utility module for CSV file handling.
    - `data/`: Directory containing input datasets for each algorithm.
    - `results/`: Directory storing output results organized by algorithm.
        - `kmeans/`: K-Means clustering results.
        - `logistic_regression/`: Logistic Regression model outputs and predictions.
        - `pca/`: PCA analysis results including eigenvalues and reconstructions.
    - `Cargo.toml`: Rust project manifest with dependencies.

- `python_code/`: Contains Python scripts and logs related to data visualization.
  - `log.txt`: Log file for tracking the execution of Python scripts.
  - `plots.py`: Main script for generating plots.
  - `requirements.txt`: List of Python dependencies required for the project.
  - `plots/`: Directory containing subdirectories for different algorithms.
    - `kmeans/`: Contains images and plots related to the K-Means algorithm.
    - `logistic_regression/`: Contains images and plots related to Logistic Regression.
    - `pca/`: Contains images and plots related to Principal Component Analysis.

> [!NOTE]
> The `report` directory and `questions.pdf` have already been addressed in this project's [main README.md](./../README.md).
