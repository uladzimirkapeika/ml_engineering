import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, hp, STATUS_OK, space_eval
import mlflow
import mlflow.sklearn

os.environ["HYPEROPT_FMIN_SEED"] = "1"
SERVICE_NAME = "server"
mlflow.set_tracking_uri(f"http://{SERVICE_NAME}:7777")


def calculate_metrics(y_pred, y_train):
    """Calculate classification metrics: Accuracy, precision and recall"""
    acc = np.mean(y_pred == y_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    return acc, spec, sens


df_train = pd.read_csv("./data/data_processed_train.csv")
y_train = df_train["target"]
vectorizer = TfidfVectorizer(max_features=50)
features = vectorizer.fit_transform(df_train["text"])

search_space = hp.choice(
    "classifier_type",
    [
        {
            "type": "svm",
            "C": hp.lognormal("SVM_C", 0, 1.0),
            "kernel": hp.choice("kernel", ["linear", "rbf"]),
        },
        {
            "type": "rf",
            "max_depth": hp.quniform("max_depth", 2, 5, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        },
        {
            "type": "logreg",
            "C": hp.lognormal("LR_C", 0, 0.5),
            "solver": hp.choice("solver", ["liblinear", "lbfgs"]),
        },
    ],
)


def objective(params):
    """Objective function for hyperopt"""
    classifier_type = params["type"]
    del params["type"]
    if classifier_type == "svm":
        clf = SVC(**params)
    elif classifier_type == "rf":
        clf = RandomForestClassifier(**params)
    elif classifier_type == "logreg":
        clf = LogisticRegression(**params)
    params["model"] = classifier_type
    y_pred = cross_val_predict(clf, features, y_train)
    acc, spec, sens = calculate_metrics(y_pred, y_train)
    mlflow.set_experiment("Model_selection")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("specificity", spec)
        mlflow.log_metric("sensitivity", sens)
    return {"loss": -acc, "status": STATUS_OK}


best_result = fmin(fn=objective, space=search_space, max_evals=30)

params = space_eval(search_space, best_result)
classifier_type_best_model = params["type"]
del params["type"]
if classifier_type_best_model == "svm":
    clf_best_model = SVC(**params)
elif classifier_type_best_model == "rf":
    clf_best_model = RandomForestClassifier(**params)
elif classifier_type_best_model == "logreg":
    clf_best_model = LogisticRegression(**params)
params["model"] = classifier_type_best_model

y_pred_best_model = cross_val_predict(clf_best_model, features, y_train)
acc_best_model, spec_best_model, sens_best_model = calculate_metrics(
    y_pred_best_model, y_train
)

model = clf_best_model.fit(features, y_train)

mlflow.set_experiment("Best_model")
with mlflow.start_run():
    mlflow.log_metric("accuracy", acc_best_model)
    mlflow.log_metric("specificity", spec_best_model)
    mlflow.log_metric("sensitivity", sens_best_model)
    mlflow.log_params(best_result)
    mlflow.sklearn.log_model(model, "best_model")
    mlflow.sklearn.log_model(vectorizer, "simple_vectorizer")
