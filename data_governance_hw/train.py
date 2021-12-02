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



vectorizer_params = {
    "max_features": 220,
    "ngram_range": (1, 2),
    "use_idf": True,
}
lr_model_params = {
    "class_weight": "balanced",
    "solver": "liblinear",
}


def prepare_data(df):
    """Create TD-IDF embeddings"""
    vectorizer = TfidfVectorizer(**vectorizer_params)
    features = vectorizer.fit_transform(df["text"])
    return vectorizer, features


def train_model(X, y):
    """Train model and calculate cross validation predictions"""
    model = RandomForestClassifier(max_depth=5, random_state=0)
    y_pred = cross_val_predict(model, X, y, cv=5)
    model.fit(X, y)
    return model, y_pred


def calculate_metrics(y_pred, y_train):
    """Calculate classification metrics: Accuracy, precision and recall"""
    acc = np.mean(y_pred == y_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    return acc, spec, sens


def calculate_permute_importance(model, X, y):
    """Calculate permute importance"""
    model.fit(X, y)
    results = permutation_importance(model, X.toarray(), y, scoring="accuracy")
    importance = results.importances_mean

    return importance


df_train = pd.read_csv("data_processed_train.csv")
vectorizer, X_train = prepare_data(df_train)
y_train = df_train["target"]

final_model, y_predictions = train_model(X_train, y_train)

accuracy, specificity, sensitivity = calculate_metrics(y_predictions, y_train)
with open("metrics.json", "w") as outfile:
    json.dump(
        {"accuracy": accuracy, "specificity": specificity, "sensitivity": sensitivity},
        outfile,
    )
permutate_importance = calculate_permute_importance(final_model, X_train, y_train)
ind = np.argpartition(permutate_importance, -10)[-10:]
top_10_words = np.array(vectorizer.get_feature_names())[ind]
sns.set(rc={'figure.figsize':(18, 12)})
ax = sns.barplot(x=top_10_words, y=permutate_importance[ind],
            label='Total', color='b', edgecolor='w')
ax.tick_params(axis='x', rotation=45)
ax.set(xlabel='Word', ylabel='Permutate importance')
sns.set_color_codes('muted')
ax.figure.savefig("permutate_feature_importance_top_10.png", dpi=80)
