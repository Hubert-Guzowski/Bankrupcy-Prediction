import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from bankruptcy_model.utils.data_loading import load_dataset_by_year
from bankruptcy_model.utils.data_prep import change_class_values_to_binary


def prepare_data_for_pca():
    df = load_dataset_by_year(5)
    df = change_class_values_to_binary(df)
    features_bankruptcy = list(df.columns.values)
    features_bankruptcy.remove('bankrupt')
    y_header = ['bankrupt']
    x_bakruptcy = df.loc[:, features_bankruptcy].values
    y_bankruptcy = df.loc[:, y_header].values
    # y_bankruptcy = change_class_values_to_binary(y_bankruptcy)
    x_bakruptcy = StandardScaler().fit_transform(x_bakruptcy)

    x_bakruptcy_standarize = pd.DataFrame(data=x_bakruptcy, columns=features_bankruptcy)

    print(x_bakruptcy_standarize.head(15))

    x_bakruptcy_standarize.fillna(x_bakruptcy_standarize.mean(), inplace=True)
    pca_bankruptcy = PCA(n_components=2)
    principalComponents_bankruptcy = pca_bankruptcy.fit_transform(x_bakruptcy_standarize)
    principalDf_bankruptcy = pd.DataFrame(data=principalComponents_bankruptcy, columns=['principal component 1',
                                                                                        'principal component 2'])
    finalDf_bankruptcy = pd.concat([principalDf_bankruptcy, df[['bankrupt']]], axis=1)
    print(finalDf_bankruptcy.head(110))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)


    bankruptcy_targets = ['1', '0']
    colors = ['r', 'g']
    for target, color in zip(bankruptcy_targets, colors):
        indicesToKeep = finalDf_bankruptcy['bankrupt'] == target
        ax.scatter(finalDf_bankruptcy.loc[indicesToKeep, 'principal component 1']
                   , finalDf_bankruptcy.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(bankruptcy_targets)
    ax.grid()
    plt.savefig('foo.png')


