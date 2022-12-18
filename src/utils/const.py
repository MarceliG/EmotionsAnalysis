import os

# Directories
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "preprocessed")

# Paths
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.pickle")

# Functions
def makedir(path: str) -> str:
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return path
