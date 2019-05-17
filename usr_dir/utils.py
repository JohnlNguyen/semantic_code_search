from pathlib import Path

import os
import pandas as pd
import tensorflow as tf
from six import StringIO

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators.function_docstring import GithubFunctionDocstring
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators.extract_raw_data import extract_data
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

def write_to_file(file_name, data):
    with open(file_name, 'w+') as f:
        for v in data:
            f.write(v + "\n")

def generate_vocab(tmp_dir, extracted_files):
    if self.vocab_type != text_problems.VocabType.TOKEN:
        tf.logging.info(
            "Not generating vocab file, vocab type is not token")
        return

    vocab_file = os.path.join(tmp_dir, "vocab.semantic_search.tokens")
    if tf.gfile.Exists(vocab_file):
        tf.logging.info("Skipping vocab generation, vocab file exists")
        return

    vocab = []
    for file in extracted_files:
        file_path = os.path.join(tmp_dir, file)
        assert tf.gfile.Exists(file_path)
        df = pd.read_json(file_path)
        for row in df[['intent_tokens', 'snippet_tokens']].itertuples():
            vocab.extend(row.intent_tokens)
            vocab.extend(row.snippet_tokens)
    vocab = set(vocab)
    write_to_file(file_name, vocab)


def tokenize_code(cls, text: str):
    "A very basic procedure for tokenizing code strings."
    return RegexpTokenizer(r'\w+').tokenize(text)