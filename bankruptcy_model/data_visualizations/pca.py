import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from bankruptcy_model.utils.data_loading import load_dataset_by_year
from bankruptcy_model.utils.data_prep import change_class_values_to_binary


def prepare_data_for_pca():
    df = load_dataset_by_year(5)
    df = change_class_values_to_binary(df)
    df = remove_outliers(df, df.iloc[: , :-1], 5)
    features_bankruptcy = list(df.columns.values)
    features_bankruptcy.remove('bankrupt')
    y_header = ['bankrupt']
    x_bakruptcy = df.loc[:, features_bankruptcy].values
    x_bakruptcy = StandardScaler().fit_transform(x_bakruptcy)

    x_bakruptcy_standarize = pd.DataFrame(data=x_bakruptcy, columns=features_bankruptcy)

    x_bakruptcy_standarize.fillna(x_bakruptcy_standarize.mean(), inplace=True)
    pca_bankruptcy = PCA(n_components=2)
    principalComponents_bankruptcy = pca_bankruptcy.fit_transform(x_bakruptcy_standarize)
    principalDf_bankruptcy = pd.DataFrame(data=principalComponents_bankruptcy, columns=['principal component 1',
                                                                                        'principal component 2'])
    df = df.reset_index(drop=True)
    finalDf_bankruptcy = pd.concat([principalDf_bankruptcy, df[['bankrupt']]], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)

    bankruptcy_targets = [1, 0]
    colors = ['r', 'g']
    for target, color in zip(bankruptcy_targets, colors):
        indices_to_keep = (finalDf_bankruptcy['bankrupt'] == target)
        ax.scatter(finalDf_bankruptcy.loc[indices_to_keep, 'principal component 1']
                   , finalDf_bankruptcy.loc[indices_to_keep, 'principal component 2']
                   , c=color)
    ax.legend(bankruptcy_targets)
    ax.grid()
    plt.savefig('foo.png')


def remove_outliers(df, out_cols, T=1.5, verbose=True):
    new_df = df.copy()
    for c in out_cols[:-1]:
        q1 = new_df[c].quantile(.25)
        q3 = new_df[c].quantile(.75)
        col_iqr = q3 - q1
        col_max = q3 + T * col_iqr
        col_min = q1 - T * col_iqr
        filtered_df = new_df[(new_df[c] <= col_max) & (new_df[c] >= col_min)]
        if verbose:
            n_out = new_df.shape[0] - filtered_df.shape[0]
            print(f" Columns {c} had {n_out} outliers removed")
        new_df = filtered_df
    return new_df


