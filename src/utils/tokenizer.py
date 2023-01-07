import os
import pickle

import pandas as pd
import tensorflow as tf
from utils.const import PREPROCESSED_DATA_DIR, TOKENIZER_PATH


def load_tokenizer() -> tf.keras.preprocessing.text.Tokenizer:
    if not os.path.exists(TOKENIZER_PATH):
        create_tokenizer()

    with open(TOKENIZER_PATH, "rb") as handle:
        return pickle.load(handle)


def create_tokenizer() -> None:
    df = pd.read_excel(os.path.join(PREPROCESSED_DATA_DIR, "train.xlsx"))
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df["text"])

    with open(TOKENIZER_PATH, "wb") as handle:
        pickle.dump(tokenizer, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    load_tokenizer()
