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

from usr_dir.utils import read_from_file

_CONALA_TRAIN_DATASETS = [
    [
        "gs://conala/",
        ("train/conala-train-bpe-seperated.intent",
         "train/conala-train-bpe-seperated.code")
    ],
    [
        "gs://conala/",
        ("mined/conala-train-mined.intent", "mined/conala-train-mined.code")
    ],
]


@registry.register_problem
class SemanticSearch(text_problems.Text2TextProblem):
    """

    """

    def __init__(self, was_reversed=False, was_copy=False):
        super(SemanticSearch, self).__init__(
            was_reversed=False, was_copy=False)

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def base_url(self):
        return "gs://conala"

    @property
    def test_file(self):
        return '{}/{}'.format(self.base_url, "conala-test.json"), "conala-test.json"

    @property
    def file_names(self):
        return [
            "conala-mined.jsonl",
            "conala-train.json"
        ]

    @property
    def pair_files_list(self):
        """
        This function returns a list of (url, file name) pairs
        """
        return [
            ('{}/{}'.format(self.base_url, name),
             name)
            for name in self.file_names
        ]

    @property
    def is_generate_per_split(self):
        return True

    @property
    def approx_vocab_size(self):
        return 2 ** 14  # ~16

    @property
    def max_samples_for_vocab(self):
        return int(3.5e5)

    @property
    def oov_token(self):
        return "UNK"

    @classmethod
    def github_data(cls, data_dir, tmp_dir, dataset_split):
        """
        Using data from function_docstring problem
        """
        github = GithubFunctionDocstring()
        return github.generate_samples(data_dir, tmp_dir, dataset_split)

    def maybe_download_conala(self, tmp_dir):
        all_files = [
            generator_utils.maybe_download(tmp_dir, file_name, uri)
            for uri, file_name in self.pair_files_list
        ]
        return all_files

    def maybe_split_data(self, tmp_dir, extracted_files, use_mined=True):

        train_file = os.path.join(
            tmp_dir, 'conala-joined-prod-train.json' if use_mined else 'conala-prod-train.json')
        valid_file = os.path.join(
            tmp_dir, 'conala-joined-prod-valid.json' if use_mined else 'conala-prod-valid.json')

        if tf.gfile.Exists(train_file) or tf.gfile.Exists(valid_file):
            tf.logging.info("Not splitting, file exists")
        else:
            if use_mined:
                df = self.join_mined_and_train(tmp_dir, extracted_files)
            else:
                train_path = os.path.join(tmp_dir, 'conala-train.json.prod')
                assert tf.gfile.Exists(train_path)
                df = pd.read_json(train_path)

            train, valid = train_test_split(
                df, test_size=0.10, random_state=42)
            train[['intent_tokens', 'snippet_tokens']].to_json(train_file)
            valid[['intent_tokens', 'snippet_tokens']].to_json(valid_file)
        return train_file, valid_file

    def join_mined_and_train(self, tmp_dir, extracted_files):
        df = pd.DataFrame([])
        for extracted_file in extracted_files:
            if 'test' not in extracted_file:
                file_path = os.path.join(tmp_dir, extracted_file)
                df = df.append(pd.read_json(file_path),
                               ignore_index=True, sort=False)
        return df

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """A generator to return data samples.Returns the data generator to return.


        Args:
          data_dir: A string representing the data directory.
          tmp_dir: A string representing the temporary directory and is¬
                  used to download files if not already available.
          dataset_split: Train, Test or Eval.

        Yields:
          Each element yielded is of a Python dict of the form
            {"inputs": "STRING", "targets": "STRING"}
        """
        extracted_files, train_filename, valid_filename = self.process_files(
            tmp_dir)

        if dataset_split == problem.DatasetSplit.TRAIN:
            df = pd.read_json(train_filename)
            for row in df.itertuples():
                yield self.get_row(row)
        elif dataset_split == problem.DatasetSplit.EVAL:
            df = pd.read_json(valid_filename)
            for row in df.itertuples():
                yield self.get_row(row)

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC,
            metrics.Metrics.APPROX_BLEU
        ]

    def get_row(self, row):
        return {"inputs": " ".join(row.intent_tokens),
                "targets": " ".join(row.snippet_tokens)}

    def process_files(self, tmp_dir):
        self.maybe_download_conala(tmp_dir)
        extracted_files = extract_data(tmp_dir, False)
        train_filename, valid_filename = self.maybe_split_data(
            tmp_dir, extracted_files, use_mined=False)
        return extracted_files, train_filename, valid_filename


@registry.register_problem
class SemanticSearchAst(SemanticSearch):
    """
    Structure this problem as a translate problem
    """

    @property
    def is_generate_per_split(self):
        return False

    @property
    def vocab_type(self):
        return text_problems.VocabType.SUBWORD

    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        extracted_files, train_filename, valid_filename = self.process_files(
            tmp_dir)

        intent_path = os.path.join(tmp_dir, 'conala-train.json.prod')
        ast_path = os.path.join(tmp_dir, 'conala-train-ast.txt')
        assert tf.gfile.Exists(intent_path) and tf.gfile.Exists(ast_path)

        intents = pd.read_json(intent_path).intent_tokens
        ast_nodes = read_from_file(ast_path)

        for intent_tokens, ast_node in zip(intents, ast_nodes):
            yield {"inputs": " ".join(intent_tokens), "targets": ast_node}



@registry.register_problem
class SemanticSearchBpe(text_problems.Text2TextProblem):
    """

    """

    def __init__(self, was_reversed=False, was_copy=False):
        super(SemanticSearchBpe, self).__init__(
            was_reversed=False, was_copy=False)

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN
    
    @property
    def vocab_filename(self):
        return "conala.vocab"

    @property
    def test_file(self):
        return '{}/{}'.format(self.base_url, "conala-test.json"), "conala-test.json"

    @property
    def pair_files_list(self):
        """
        This function returns a list of (url, file name) pairs
        """
        return [
            ('{}/{}'.format(self.base_url, name),
             name)
            for name in self.file_names
        ]

    @property
    def is_generate_per_split(self):
        return True

    @property
    def approx_vocab_size(self):
        return 2 ** 14  # ~16

    @property
    def max_samples_for_vocab(self):
        return int(3.5e5)

    @property
    def oov_token(self):
        return "UNK"

    @classmethod
    def github_data(cls, data_dir, tmp_dir, dataset_split):
        """
        Using data from function_docstring problem
        """
        github = GithubFunctionDocstring()
        return github.generate_samples(data_dir, tmp_dir, dataset_split)

    def maybe_download_conala(self, tmp_dir):
        all_files = [
            generator_utils.maybe_download(tmp_dir, file_name, uri)
            for uri, file_name in self.pair_files_list
        ]
        return all_files

    def maybe_split_data(self, tmp_dir, extracted_files, use_mined=True):

        train_file = os.path.join(
            tmp_dir, 'conala-joined-prod-train.json' if use_mined else 'conala-prod-train.json')
        valid_file = os.path.join(
            tmp_dir, 'conala-joined-prod-valid.json' if use_mined else 'conala-prod-valid.json')

        if tf.gfile.Exists(train_file) or tf.gfile.Exists(valid_file):
            tf.logging.info("Not splitting, file exists")
        else:
            if use_mined:
                df = self.join_mined_and_train(tmp_dir, extracted_files)
            else:
                train_path = os.path.join(tmp_dir, 'conala-train.json.prod')
                assert tf.gfile.Exists(train_path)
                df = pd.read_json(train_path)

            train, valid = train_test_split(
                df, test_size=0.10, random_state=42)
            train[['intent_tokens', 'snippet_tokens']].to_json(train_file)
            valid[['intent_tokens', 'snippet_tokens']].to_json(valid_file)
        return train_file, valid_file

    def join_mined_and_train(self, tmp_dir, extracted_files):
        df = pd.DataFrame([])
        for extracted_file in extracted_files:
            if 'test' not in extracted_file:
                file_path = os.path.join(tmp_dir, extracted_file)
                df = df.append(pd.read_json(file_path),
                               ignore_index=True, sort=False)
        return df

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """A generator to return data samples.Returns the data generator to return.


        Args:
          data_dir: A string representing the data directory.
          tmp_dir: A string representing the temporary directory and is¬
                  used to download files if not already available.
          dataset_split: Train, Test or Eval.

        Yields:
          Each element yielded is of a Python dict of the form
            {"inputs": "STRING", "targets": "STRING"}
        """
        code_file = os.path.join(data_dir, "train/conala-train-bpe-untagged.code")
        intent_file = os.path.join(data_dir, "train/conala-train-bpe-untagged.intent")
        with open(code_file, 'rb', encoding='utf-8') as fc:
            with open(intent_file, 'rb', encoding='utf-8') as fi:
                for i, line in fc:
                    yield {"inputs": fi.readline(), "targets": line}

        code_file = os.path.join(data_dir, "mined/conala-train-mined-bpe-untagged.code")
        intent_file = os.path.join(data_dir, "mined/conala-train-mined-bpe-untagged.intent")
        with open(code_file, 'rb', encoding='utf-8') as fc:
            with open(intent_file, 'rb', encoding='utf-8') as fi:
                for i, line in fc:
                    yield {"inputs": fi.readline(), "targets": line}

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC,
            metrics.Metrics.APPROX_BLEU
        ]
        