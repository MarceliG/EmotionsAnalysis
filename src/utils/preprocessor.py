import os
import nltk
import string
import pandas as pd
from const import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, makedir

COLUMNS = [
    "joy",
    "trust",
    "anticipation",
    "surprise",
    "fear",
    "sadness",
    "disgust",
    "anger",
    "valence",
    "arousal",
]


def preprocess() -> None:
    download_nltk_packages()

    for dataset in os.listdir(RAW_DATA_DIR):
        df = pd.read_excel(os.path.join(RAW_DATA_DIR, dataset))

        # Punctuation removal
        df["text"] = df["text"].apply(lambda text: remove_punctuation(text))
        # Lowering the text
        df["text"] = df["text"].apply(lambda text: text.lower())
        # Stop word removal
        df["text"] = df["text"].apply(lambda text: remove_stopwords(text))
        # Lemmatization
        df["text"] = df["text"].apply(lambda text: lemmatize(text))
        # Tokens to sentence
        df["text"] = df["text"].apply(lambda text: " ".join(text))

        # Annotators merge
        df_x = df[["text"] + [f"{column}_x" for column in COLUMNS]]
        df_y = df[["text"] + [f"{column}_y" for column in COLUMNS]]
        df_x.columns = ["text"] + COLUMNS
        df_y.columns = ["text"] + COLUMNS
        df = pd.concat([df_x, df_y])

        df_to_excel(df, os.path.join(PREPROCESSED_DATA_DIR, dataset))


def lemmatize(text: list[str]) -> list[str]:
    def lemmatizer(
        wl: nltk.stem.WordNetLemmatizer, text: list[str], pos: str
    ) -> list[str]:
        return [wl.lemmatize(word, pos=pos) for word in text]

    wl = nltk.stem.WordNetLemmatizer()
    lemmatized_text = lemmatizer(wl, text, "n")
    lemmatized_text = lemmatizer(wl, lemmatized_text, "v")
    lemmatized_text = lemmatizer(wl, lemmatized_text, "a")
    lemmatized_text = lemmatizer(wl, lemmatized_text, "r")
    lemmatized_text = lemmatizer(wl, lemmatized_text, "s")
    return lemmatized_text


def remove_stopwords(text: str) -> list[str]:
    stopwords = nltk.corpus.stopwords.words("english")
    tokenized_text = nltk.tokenize.word_tokenize(text)
    return [word for word in tokenized_text if word not in stopwords]


def remove_punctuation(text: str) -> str:
    return "".join([char for char in text if char not in string.punctuation])


def download_nltk_packages() -> None:
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("stopwords")


def df_to_excel(df: pd.DataFrame, path: str) -> None:
    makedir(path)
    df.to_excel(path, index=False)


if __name__ == "__main__":
    preprocess()
