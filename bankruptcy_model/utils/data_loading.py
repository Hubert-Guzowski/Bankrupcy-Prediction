import os

from scipy.io import arff
import pandas as pd


from django.conf import settings


def load_dataset_by_year(year_number) -> pd.DataFrame:
    file_path = os.path.join(settings.BASE_DIR, 'data', str(year_number) + 'year.arff')
    data = arff.loadarff(file_path)
    return pd.DataFrame(data[0])


def combine_datasets_into_one_dataset() -> pd.DataFrame:
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
    return result
