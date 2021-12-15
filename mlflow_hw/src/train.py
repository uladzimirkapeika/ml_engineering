import json
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials, space_eval
import os
import pickle
from mlflow.tracking import MlflowClient

# os.environ['HYPEROPT_FMIN_SEED'] = "1"
mlflow.set_tracking_uri("http://localhost:7777")


def calculate_metrics(y_pred, y_train):
    """Calculate classification metrics: Accuracy, precision and recall"""
    acc = np.mean(y_pred == y_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    return acc, spec, sens


client = MlflowClient()
experiment_id = client.create_experiment("Test N1")
experiment = client.get_experiment(experiment_id)
print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

df_train = pd.read_csv("data_processed_train.csv")
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
    with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("specificity", spec)
        mlflow.log_metric("sensitivity", sens)
    return {"loss": -acc, "status": STATUS_OK}


best_result = fmin(fn=objective, space=search_space, max_evals=3)

params = space_eval(search_space, best_result)
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

model = clf.fit(features, y_train)
params["model"] = classifier_type

with mlflow.start_run(experiment_id=experiment_id, run_name="best_model") as run:
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("specificity", spec)
    mlflow.log_metric("sensitivity", sens)
    mlflow.log_params(best_result)
    mlflow.sklearn.log_model(model, "best_model")
    mlflow.sklearn.log_model(vectorizer, "simple_vectorizer")

client = MlflowClient()
# local_dir = "volume/artifact_downloads"
# if not os.path.exists(local_dir):
#    os.mkdir(local_dir)
# local_path = client.download_artifacts(run.info.run_id, "model", local_dir)
# print("Artifacts downloaded in: {}".format(local_path))
# print("Artifacts: {}".format(os.listdir(local_path)))
