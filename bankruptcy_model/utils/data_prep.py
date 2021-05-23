import gc
import pickle

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import plot_precision_recall_curve
# from scikitplot.metrics import plot_precision_recall
import matplotlib.pyplot as plt

from bankruptcy_model.utils.data_loading import load_dataset_by_year


def change_class_values_to_binary(df):
    df.loc[df['class'].astype(str).str.contains("0"), 'bankrupt'] = 0
    df.loc[df['class'].astype(str).str.contains("1"), 'bankrupt'] = 1
    df = df.drop(['class'], axis=1)

    return df


def perform_smote(df):
    # df.dropna(inplace=True)
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp_data = imp.fit_transform(df)
    # imp_data_df = pd.DataFrame(imp_data, index=df.index, columns=df.columns)
    #
    # X = imp_data_df[imp_data_df.columns.difference(['bankruptcy_after_years', 'class'])]
    # Y = imp_data_df['bankruptcy_after_years']
    #
    # del imp_data, imp_data_df
    # gc.collect()
    #
    # sm = SMOTE(random_state=111)
    # X_smote, y_smote = sm.fit_resample(X, Y)
    df.dropna(inplace=True)
    all_frs = df[df.columns.difference(['bankruptcy_after_years', 'class'])]  # with NaNs
    label = df['bankruptcy_after_years']

    sm = SMOTE(random_state=111)
    X_smote, y_smote = sm.fit_resample(all_frs, label)
    return X_smote, y_smote


def create_model(df):
    X_smote, y_smote = perform_smote(df)
    param_grid = {
        'max_depth': [3, 5, 7, 10]
    }
    CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
    CV_rfc.fit(X_smote, y_smote)

    print(CV_rfc.best_params_)

    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.3)

    models = [(RandomForestClassifier(max_depth=10), X_train_smote, y_train_smote,
               'random_forest_SMOTE_no_nan.pkl')]

    for clr, frs, label, out_dir in models:
        save_model(clr, frs, label, out_dir)

    test_models = [('random_forest_SMOTE_no_nan.pkl', X_test_smote, y_test_smote, 'Random forest + SMOTE')]

    for in_dir, X_test, y_test, title in test_models:
        validate(in_dir, X_test, y_test, title)


def save_model(model, frs, label, out_dir):
    model.fit(frs, label)
    pickle.dump(model, open(out_dir, 'wb'))


def validate(in_dir, X_test, y_test, title):
    pass
    # model = pickle.load(open(in_dir, 'rb'))
    # y_probas = model.predict_proba(X_test)
    # plot_precision_recall_curve(y_test, y_probas,
    #                             title=str('Precision-recall curve micro-averaged over all classes for ' + title))
    #
    # plt.savefig(str(title + '.jpg'))
    # # plt.show()