import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

nltk.download("punkt")
nltk.download("stopwords")


def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
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


df_train = pd.read_csv("train.csv", sep=",")
snowball = SnowballStemmer(language="english")

df_train["text"] = df_train["text"].apply(tokenize_sentence)
df_train.to_csv("data_processed_train.csv")
