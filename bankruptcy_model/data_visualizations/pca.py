import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bankruptcy_model.utils.data_loading import load_dataset_by_year
from bankruptcy_model.utils.data_prep import change_class_values_to_binary


def prepare_data_for_pca(year):
    df = load_dataset_by_year(year)
    df = change_class_values_to_binary(df)
    df = remove_outliers(df, df.iloc[:, :-1], 5)
    features_bankruptcy = list(df.columns.values)
    features_bankruptcy.remove('bankrupt')
    x_bankruptcy = df.loc[:, features_bankruptcy].values
    x_bankruptcy = StandardScaler().fit_transform(x_bankruptcy)

    x_bankruptcy_standardized = pd.DataFrame(data=x_bankruptcy, columns=features_bankruptcy)

    x_bankruptcy_standardized.fillna(x_bankruptcy_standardized.mean(), inplace=True)
    pca_bankruptcy = PCA(n_components=2)
    principal_components_bankruptcy = pca_bankruptcy.fit_transform(x_bankruptcy_standardized)
    principal_df_bankruptcy = pd.DataFrame(data=principal_components_bankruptcy,
                                          columns=['principal component 1', 'principal component 2'])
    df = df.reset_index(drop=True)
    final_df_bankruptcy = pd.concat([principal_df_bankruptcy, df[['bankrupt']]], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)

    bankruptcy_targets = [1, 0]
    colors = ['r', 'g']
    for target, color in zip(bankruptcy_targets, colors):
        indices_to_keep = (final_df_bankruptcy['bankrupt'] == target)
        ax.scatter(final_df_bankruptcy.loc[indices_to_keep, 'principal component 1']
                   , final_df_bankruptcy.loc[indices_to_keep, 'principal component 2']
                   , c=color)
    ax.legend(bankruptcy_targets)
    ax.grid()

    plt.savefig("pca.png")


def remove_outliers(df, out_cols, t=1.5, verbose=True):
    new_df = df.copy()
    for c in out_cols[:-1]:
        q1 = new_df[c].quantile(.25)
        q3 = new_df[c].quantile(.75)
        col_iqr = q3 - q1
        col_max = q3 + t * col_iqr
        col_min = q1 - t * col_iqr
        filtered_df = new_df[(new_df[c] <= col_max) & (new_df[c] >= col_min)]
        if verbose:
            n_out = new_df.shape[0] - filtered_df.shape[0]
            print(f" Columns {c} had {n_out} outliers removed")
        new_df = filtered_df
    return new_df


