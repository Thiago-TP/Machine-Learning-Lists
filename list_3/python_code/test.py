import os
import pandas as pd

# import seaborn as sns


def main():
    root = "./data/multiclass_classification/"
    with open(
        root
        + "pads-parkinsons-disease-smartwatch-dataset-1.0.0/"
        + "preprocessed/"
        + "movement/"
        + "001_ml.bin",
        "rb",
    ) as f:
        content = f.read()
        print(content.decode(encoding="utf-8", errors="ignore"))
    # df = pd.read_csv(root + "ALAMEDA_PD_tremor_dataset.csv").drop(
    #     columns=[
    #         "start_timestamp",
    #         "end_timestamp",
    #         "subject_id",
    #         "PC1_neg_rt",
    #         "Magnitude_fft_dom_freq",
    #         "Magnitude_fft_pw_ar_dom_freq",
    #     ]
    # )
    # corr = df.corr()
    # mask = corr > 0.9
    # high_corr = corr[mask]
    # col_to_filter = ~high_corr.any()
    # clean_df = df[high_corr.columns[col_to_filter]]

    # print(corr.to_string())
    # print(clean_df.to_string())


if __name__ == "__main__":
    main()
