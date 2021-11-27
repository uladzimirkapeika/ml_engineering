import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score

df_train = pd.read_csv('data_processed_train.csv')
df_test = pd.read_csv('data_processed_test.csv')

vectorizer_params = {
    "max_features": 1000,
    #"max_features": None,
    #"ngram_range": (1, 2),
    #"min_df": 0,
    #"max_df": 100,
    #"use_idf": False,
    #"decode_error": "replace",
    #"sublinear_tf": True,
    #"analyzer": "char"
}
lr_model_params = {
    #"class_weight": "balanced",
    #"class_weight": None,
    #"class_weight": {1: 1, 0: 1/class_ratio},
    #"random_state": 0,
    #"Cs": 5,
    #"penalty": "none",
    #"penalty": "elasticnet",
    "solver": "liblinear",
    #"l1_ratio": 0.5,
    #"max_iter": 10000,
    #"cv": 10
}

vectorizer = TfidfVectorizer(**vectorizer_params)
features = vectorizer.fit_transform(df_train["text"])

X_train = features
y_train = df_train['target']

model = lm.LogisticRegressionCV(**lr_model_params)
y_pred = cross_val_predict(model, X_train, y_train, cv=5)

#test_features = vectorizer.transform(df_test["text"])
#y_test = df_test['target']
#y_pred = model.predict(test_features)

acc = np.mean(y_pred == y_train)
tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)

with open("metrics.json", 'w') as outfile:
    json.dump({"accuracy": acc, "specificity": specificity, "sensitivity": sensitivity}, outfile)

model.fit(X_train, y_train)
print(X_train.toarray().shape)
results = permutation_importance(model, X_train.toarray(), y_train, scoring='accuracy')
importance = results.importances_mean

for i,v in enumerate(importance[0:10]):
	print('Feature: %0d, Score: %.5f' % (i,v))

ax = pyplot.bar([x for x in range(len(importance[0:10]))], importance)
pyplot.savefig("permutate_feature_importance_top_10.png", dpi=80)