import pandas as pd

from bankruptcy_model.utils.data_loading import load_dataset_by_year


def change_class_values_to_binary(df):
    df.loc[df['class'].astype(str).str.contains("0"), 'bankrupt'] = 0
    df.loc[df['class'].astype(str).str.contains("1"), 'bankrupt'] = 1
    df = df.drop(['class'], axis=1)

    return df
