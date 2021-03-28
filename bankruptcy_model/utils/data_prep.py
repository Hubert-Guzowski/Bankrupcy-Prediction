import pandas as pd

from bankruptcy_model.utils.data_loading import load_dataset_by_year


def combine_datasets_into_one_dataset():
    df_1 = load_dataset_by_year(1)
    df_1['year'] = 1
    df_2 = load_dataset_by_year(2)
    df_2['year'] = 2
    df_3 = load_dataset_by_year(3)
    df_3['year'] = 3
    df_4 = load_dataset_by_year(4)
    df_4['year'] = 4
    df_5 = load_dataset_by_year(5)
    df_5['year'] = 5
    frames = [df_1, df_2, df_3, df_4, df_5]
    result = pd.concat(frames)
    print(result)

    return result


def change_class_values_to_binary(df):
    df.loc[df['class'].astype(str).str.contains("0"), 'bankrupt'] = 0
    df.loc[df['class'].astype(str).str.contains("1"), 'bankrupt'] = 1
    df = df.drop(['class'], axis=1)

    return df
