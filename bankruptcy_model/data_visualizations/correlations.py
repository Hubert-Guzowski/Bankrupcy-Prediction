import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats

from bankruptcy_model.utils.data_loading import load_dataset_by_year
from bankruptcy_model.utils.data_prep import change_class_values_to_binary


def data_prep(data):
    data = data.dropna()
    data = data_normalization(data)
    return data


def drop_outliers(df):
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = df[filtered_entries]
    return new_df


def data_normalization(data):
    normed = preprocessing.scale(data)
    normed = pd.DataFrame(normed)
    return normed


def plot_corrmatrix(data, year):
    data = load_dataset_by_year(year)
    data = change_class_values_to_binary(data)
    corr_matrix = data.corr()
    plt.figure(figsize=(20,20))
    sn.heatmap(corr_matrix, cmap='coolwarm')
    plt.show()



