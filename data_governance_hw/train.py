import pandas as pd
import numpy as np
import json
import sklearn.linear_model as lm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance
from matplotlib import pyplot


vectorizer_params = {
    "max_features": 300,
    # "max_features": None,
    # "ngram_range": (1, 2),
    # "min_df": 0,
    # "max_df": 100,
     "use_idf": False,
    # "decode_error": "replace",
    # "sublinear_tf": True,
    # "analyzer": "char"
}
lr_model_params = {
    "class_weight": "balanced",
    # "class_weight": None,
    # "class_weight": {1: 1, 0: 1/class_ratio},
    # "random_state": 0,
    # "Cs": 5,
    # "penalty": "none",
    # "penalty": "elasticnet",
    "solver": "liblinear",
     "l1_ratio": 0.5,
    # "max_iter": 10000,
    # "cv": 10
}


def prepare_data(df_train):
    vectorizer = TfidfVectorizer(**vectorizer_params)
    features = vectorizer.fit_transform(df_train["text"])
    return vectorizer, features


def train_model(X_train, y_train):
    model = lm.LogisticRegressionCV(**lr_model_params)
    y_pred = cross_val_predict(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    return model, y_pred


def calculate_metrics(y_pred, y_train):
    acc = np.mean(y_pred == y_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return acc, specificity, sensitivity


def calculate_permute_importance(model, X_train, y_train):
    model.fit(X_train, y_train)
    results = permutation_importance(
        model, X_train.toarray(), y_train, scoring="accuracy"
    )
    importances = results.importances_mean

    return importances


df_train = pd.read_csv("data_processed_train.csv")
_, X_train = prepare_data(df_train)
y_train = df_train["target"]

model, y_pred = train_model(X_train, y_train)

acc, specificity, sensitivity = calculate_metrics(y_pred, y_train)
with open("metrics.json", "w") as outfile:
    json.dump(
        {"accuracy": acc, "specificity": specificity, "sensitivity": sensitivity},
        outfile,
    )
importance = calculate_permute_importance(model, X_train, y_train)

for i, v in enumerate(importance[0:10]):
    print("Feature: %0d, Score: %.5f" % (i, v))

ax = pyplot.bar([x for x in range(len(importance[0:10]))], importance[0:10])
pyplot.savefig("permutate_feature_importance_top_10.png", dpi=80)
