import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.io import arff
from scipy import stats


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


def plot_corrmatrix(data):
    corrMatrix = data.corr()
    plt.figure(figsize=(20,20))
    sn.heatmap(corrMatrix, cmap='coolwarm')
    plt.show()


data_year5 = arff.loadarff('5year.arff')
df_5 = pd.DataFrame(data_year5[0])
df_5.loc[df_5['class'].astype(str).str.contains("0"), 'bankrupt'] = 0
df_5.loc[df_5['class'].astype(str).str.contains("1"), 'bankrupt'] = 1
plot_corrmatrix(df_5)
