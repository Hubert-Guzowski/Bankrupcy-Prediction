import os

from scipy.io import arff
import pandas as pd

from bankruptcy_prediction.settings import BASE_DIR


def load_dataset_by_year(year_number) -> pd.DataFrame:
    file_path = os.path.join(BASE_DIR, 'data', str(year_number)+'year.arff')
    data = arff.loadarff(file_path)
    return pd.DataFrame(data[0])
