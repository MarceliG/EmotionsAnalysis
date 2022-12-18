import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.tokenizer import load_tokenizer
from utils.const import PREPROCESSED_DATA_DIR, makedir

# Model params
D = 20           # Embedding dimensionality
M = 15           # Hidden / Cell state dimensionality
EPOCHS = 10

MODEL_SAVE_PATH = os.path.join("models", "lstm")

def train():
    trn, val = load_datasets()

    # Tokenization
    tokenizer = load_tokenizer()
    tokens_trn = tokenizer.texts_to_sequences(trn["text"])
    tokens_val = tokenizer.texts_to_sequences(val["text"])
    V = len(tokenizer.word_index)                   # Vocabulary size
    T = trn["text"].map(lambda x: len(x)).max()     # Input sequence size / max sequence length

    # Padding
    data_trn = tf.keras.utils.pad_sequences(tokens_trn, T, padding="post")
    data_val = tf.keras.utils.pad_sequences(tokens_val, T, padding="post")

    model = None
    for emotion in trn.columns[1:]:
        # 
        y_trn = tf.keras.utils.to_categorical(trn[emotion])
        y_val = tf.keras.utils.to_categorical(val[emotion])
        K = y_trn.shape[1]

        # Building model
        i = tf.keras.layers.Input(shape=(T,), name="input")
        x = tf.keras.layers.Embedding(V + 1, D, name="embedding")(i)
        x = tf.keras.layers.LSTM(M, return_sequences=True, name="lstm")(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name="global-max-pooling-1d")(x)
        x = tf.keras.layers.Dense(K, activation="softmax", name="classif")(x)
        model = tf.keras.models.Model(i, x, name=emotion)

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=[
                tf.keras.metrics.Accuracy(name="accuracy"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            ]
        )

        # Training model
        r = model.fit(
            data_trn,
            y_trn,
            epochs=EPOCHS,
            validation_data=(data_val, y_val)
        )

        # Saving results
        model.save(makedir(os.path.join(MODEL_SAVE_PATH, "models", emotion)))
        np.save(makedir(os.path.join(MODEL_SAVE_PATH, "history", emotion)), r.history)

    # Saving model info
    with open(os.path.join(MODEL_SAVE_PATH, "architecture.txt"), "w") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    with open(os.path.join(MODEL_SAVE_PATH, "config.json"), "w") as f:
        json.dump({"T": T}, f)

def history() -> None:
    pass

def predict() -> None:
    pass

def evaluate() -> None:
    pass

def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_excel(os.path.join(PREPROCESSED_DATA_DIR, "train.xlsx")), pd.read_excel(os.path.join(PREPROCESSED_DATA_DIR, "valid.xlsx"))

if __name__ == "__main__":
    train()
