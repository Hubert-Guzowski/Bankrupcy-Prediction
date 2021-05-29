import gc
import pickle

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import plot_precision_recall_curve, precision_recall_curve, accuracy_score, f1_score
import matplotlib.pyplot as plt

from bankruptcy_model.utils.data_loading import load_dataset_by_year
from sklearn.preprocessing import StandardScaler


def change_class_values_to_binary(df):
    df.loc[df['class'].astype(str).str.contains("0"), 'bankrupt'] = 0
    df.loc[df['class'].astype(str).str.contains("1"), 'bankrupt'] = 1
    df = df.drop(['class'], axis=1)

    return df


def perform_smote_method(df):
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.drop(['Unnamed: 0.1'], axis=1)
    df = df.drop(['id'], axis=1)
    df = df.drop(['year'], axis=1)
    df = df.dropna()
    X = df.drop(['class', 'bankruptcy_after_years'], axis=1)
    y_class = df['class']
    y_years = df['bankruptcy_after_years']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X.values)
    X = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)

    strategy = {0: 19535, 1: 1020, 2: 1200, 3: 1070, 4: 730, 5: 300}
    sm = SMOTE(random_state=111, sampling_strategy=strategy)
    X_sm, y_sm = sm.fit_resample(X, y_years)

    return X_sm, y_sm


def perform_random_forest_model(df):
    X_sm, y_sm = perform_smote_method(df)
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("f1:", f1_score(y_test, y_pred, average=None))


def save_model(model, frs, label, out_dir):
    model.fit(frs, label)
    pickle.dump(model, open(out_dir, 'wb'))


def validate(in_dir, X_test, y_test, title):
    model = pickle.load(open(in_dir, 'rb'))
    y_probas = model.predict_proba(X_test)
    fps, tps, thresholds = precision_recall_curve(y_test, y_probas)

    plt.savefig(str(title + '.jpg'))
    # plt.show()