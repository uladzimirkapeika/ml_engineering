import pandas as pd
import numpy as np
import dask.dataframe as dd
import airflow
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from kaggle.api.kaggle_api_extended import KaggleApi

nltk.download("punkt")
nltk.download("stopwords")
snowball = SnowballStemmer(language="english")


def save_file(df, name):
    dd.to_csv('/opt/airflow/data/' + name + '.csv', sep=',', index=False)


def load_file(name):
    df = dd.read_csv("/opt/airflow/data/" + name + ".csv")
    return df


def load_data():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file('jp797498e/twitter-entity-sentiment-analysis', '/opt/airflow/data/twitter_training.csv')
    api.dataset_download_file('jp797498e/twitter-entity-sentiment-analysis', '/opt/airflow/data/twitter_validation.csv')


def clean_sentence(sentence: str, remove_stop_words: bool = True):
    """
    Clean text data
    :param sentence: raw text
    :param remove_stop_words: boolean flag specifying if stop words have to be removed
    :return: cleaned text
    """
    tokens = word_tokenize(sentence, language="english")
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in stopwords.words("english")]
    tokens = [snowball.stem(i) for i in tokens]
    return " ".join(tokens)


def sentiment_scores(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict['neg'], sentiment_dict['neu'], sentiment_dict['pos']


def process_data():
    df_train = load_file('twitter_training.csv')
    df_test = load_file('twitter_validation.csv')

    df_train = df_train.apply(clean_sentence)
    df_test = df_test.apply(clean_sentence)

    df_train['char_count'] = df_train['text'].apply(lambda x: len(x))
    df_train['words_count'] = df_train['text'].apply(lambda x: len(x.split()))
    df_train['avg_word_length'] = df_train['char_count'] / df_train['words_count']

    df_test['char_count'] = df_test['text'].apply(lambda x: len(x))
    df_test['words_count'] = df_test['text'].apply(lambda x: len(x.split()))
    df_test['avg_word_length'] = df_test['char_count'] / df_test['words_count']

    df_train[['neg_vader_score', 'neu_vader_score', 'pos_vader_score']] = df_train['text'].apply(
        lambda x: sentiment_scores(x))
    df_test[['neg_vader_score', 'neu_vader_score', 'pos_vader_score']] = df_test['text'].apply(
        lambda x: sentiment_scores(x))

    vectorizer_params = {
        "max_features": 350,
        "ngram_range": (1, 2),
        "use_idf": True,
    }
    vectorizer = TfidfVectorizer(**vectorizer_params)
    tdidf_train_features = vectorizer.fit_transform(df_train["text"])
    tdidf_test_features = vectorizer.transform(df_test["text"])

    col_names = [f'td_idf_{i}' for i in range(len(tdidf_train_features))]

    df_train[col_names] = tdidf_train_features
    df_test[col_names] = tdidf_test_features

    df_train.drop_columns(['text'], inplace=True)
    df_test.drop_columns(['text'], inplace=True)

    save_file(df_train, "temp_train.csv")
    save_file(df_test, "temp_test.csv")


def save_data():
    df_train = load_file('temp_train.csv')
    df_test = load_file('temp_test.csv')

    assert df_train.shape[0] > 0
    assert df_test.shape[0] > 0
    assert df_train.shape[1] == df_test.shape[1]

    save_file('processed_train.csv')
    save_file('processed_test.csv')


default_args = {
    'start_date': datetime(2021, 28, 12),
    'email': ['airflow_notification@thisisadummydomain.com'],
    'email_on_failure': False
}


dag = DAG('etl_pipeline',
          description='etl_pipeline',
          default_args=default_args,
          schedule_interval='@daily'
          )


with dag:
    load_data_task = PythonOperator(task_id='clean_text',
                                 python_callable=load_data)

    process_data_task = PythonOperator(task_id='process_data',
                                 python_callable=process_data)

    save_data_task = PythonOperator(task_id='save_data',
                                 python_callable=save_data)

    clean_dir_task = BashOperator(task_id='clean_dir', bash_command="cleanup.sh")

    load_data_task >> process_data_task >> save_data_task >> clean_dir_task