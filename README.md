
# Machine Learning Lists

This repository contains solutions to three machine learning programming assignments, each organized in its own directory.

## 1 Repository Structure

```
machine-learning-lists/
├── list_1/
├── list_2/
├── list_3/
└── professors_code/
```

## 2 Directories

### 2.1 `list_1/`
First assignment focusing on probability, statistics, and basic machine learning concepts. Contains implementations in Python and LaTeX report.

### 2.2 `list_2/`
Second assignment covering clustering and dimensionality reduction. Includes Python and Rust implementations with machine learning algorithms.

### 2.3 `list_3/`
Third assignment on classification methods. Features Python implementations of various classifiers with data analysis and visualization.

### 2.4 `professors_code/`
Reference implementations provided by instructors, including MATLAB examples for K-means, PCA, and logistic regression.

## 3 Common Structure (list_1, list_2, list_3)

Each assignment directory follows this pattern:

- **`README.md`** - Assignment overview and instructions
- **`python_code/`** - Python implementations, data, and results
- **`report/`** - LaTeX source files for the final report
- **`questions.pdf`** - Problem statement with assignment questions (in main directory)

> [!NOTE]
> List 2, in particular, implements machine learning algorithms in Rust, focusing the Python code entirely on plotting results.
> Therefore, on top of the usual `python_code` folder, this list also has a `rust_code` folder.

## 4 Questions PDF

The `questions.pdf` file in each assignment directory contains the complete list of problems to solve. It defines the scope and requirements for all code implementations and analyses.

## 5 Generating Report PDFs

To generate the PDFs for the reports in each list, follow these steps:

1. **Navigate to the report directory** of the respective list:
    - For list 1: `cd list_1/report`
    - For list 2: `cd list_2/report`
    - For list 3: `cd list_3/report`

2. **Compile the LaTeX files** using the following command:
    ```bash
    pdflatex report.tex
    ```
    You may need to run this command multiple times to resolve references and citations.

3. **Check for any errors** during the compilation process. If there are issues, review the `.log` file generated for details.

4. **Locate the generated PDF** in the same directory. The file will be named `report.pdf`.

5. **Repeat the process** for each list to generate their respective reports.

> [!WARNING]
> Ensure that you have a LaTeX distribution installed (e.g., TeX Live, MiKTeX) to compile the documents successfully.