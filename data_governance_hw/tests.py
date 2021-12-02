import string
import time
import pickle
import pandas as pd
import process_data
import train


def test_input_data_shape():
    """ Check input data: shape, length of text and class balance"""
    df_train = pd.read_csv("train.csv", sep=",")
    assert df_train.shape[0] > 0
    assert df_train.shape[1] > 0
    assert (0 < df_train["text"].apply(len)).all()
    assert (280 > df_train["text"].apply(len)).all()
    assert (df_train["target"].isin((0, 1))).all()
    assert 0 < sum(df_train["target"]) < df_train.shape[0]


def test_cleaning_text_function():
    """ Test the funcion for cleaning raw text"""
    test1 = """My cousin in Trinidad won’t get the vaccine cuz his friend got it & became crazy.
        His friend was weeks away from getting married, now the girl called off the wedding.
        So just pray on it & make sure you’re comfortable with ur decision, not bullied"""
    processed_text = process_data.tokenize_sentence(test1)
    assert len(set(processed_text.split()) & set(string.punctuation)) == 0
    # assert len(set(processed_text.split()) & set(stopwords.words("english"))) == 0


def test_processed_data_for_training_purposes():
    """ Test processed data. It should not contain empty text"""
    df_train = pd.read_csv("train.csv", sep=",")
    df_train["text"] = df_train["text"].apply(process_data.tokenize_sentence)
    df_train.to_csv("data_processed_train.csv")
    assert all(df_train["text"].apply(len) > 0)


def test_text_vectorizer():
    """Test TD-IDF vectorizer. It should return values from 0 to 1"""
    df_processed = pd.read_csv("data_processed_train.csv", sep=",")
    vectorizer, features = train.prepare_data(df_processed)
    features = features.toarray()
    assert features.shape[0] > 0
    assert (features <= 1).all()
    filename = "vectorizer.sav"
    pickle.dump(vectorizer, open(filename, "wb"))


def test_training_time():
    """Test training time for the model"""
    df_processed = pd.read_csv("data_processed_train.csv", sep=",")
    _, features = train.prepare_data(df_processed)
    X_train, y_train = features, df_processed["target"]
    time_before = time.time()
    model, _ = train.train_model(X_train, y_train)
    assert time.time() - time_before < 600
    filename = "model.sav"
    pickle.dump(model, open(filename, "wb"))


def test_inference_time():
    """Test inference time for the model"""
    test_text = "This is the test tweet to check the inference time"
    processed_text = process_data.tokenize_sentence(test_text)
    print(processed_text)
    filename = "model.sav"
    model = pickle.load(open(filename, "rb"))

    filename = "vectorizer.sav"
    vectorizer = pickle.load(open(filename, "rb"))

    df_temp = pd.DataFrame()
    df_temp["text"] = [processed_text]
    features = vectorizer.transform(df_temp["text"])
    time_before = time.time()
    _ = model.predict(features)
    assert time.time() - time_before < 100


def test_metrics_values():
    """Check model quality and the correctness of metrics calculation"""
    df_processed = pd.read_csv("data_processed_train.csv", sep=",")
    _, features = train.prepare_data(df_processed)
    X_train, y_train = features, df_processed["target"]
    _, y_pred = train.train_model(X_train, y_train)
    acc, specificity, sensitivity = train.calculate_metrics(y_pred, y_train)
    assert 0.5 < acc < 1
    assert 0 < specificity < 1
    assert 0 < sensitivity < 1


def test_permutation_importance_results():
    """Check calculation of permutation importance for the model"""
    df_processed = pd.read_csv("data_processed_train.csv", sep=",")
    _, features = train.prepare_data(df_processed)
    X_train, y_train = features, df_processed["target"]
    filename = "model.sav"
    model = pickle.load(open(filename, "rb"))
    pi_results = train.calculate_permute_importance(model, X_train, y_train)
    assert len(pi_results == 10)
