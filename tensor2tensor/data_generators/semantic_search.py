from pathlib import Path

import os
import pandas as pd
import tensorflow as tf
from six import StringIO

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators.function_docstring import GithubFunctionDocstring
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators.extract_raw_data import extract_data
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

@registry.register_problem
class SemanticSearch(text_problems.Text2TextProblem):
    """

    """

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
        return False

    @property
    def approx_vocab_size(self):
        return 2 ** 13

    @property
    def max_samples_for_vocab(self):
        return int(3.5e5)

    def maybe_download_conala(self, tmp_dir):
        all_files = [
            generator_utils.maybe_download(tmp_dir, file_name, uri)
            for uri, file_name in self.pair_files_list
        ]
        return all_files

    def maybe_split_data(self, tmp_dir, extracted_files):
        train_file = os.path.join(tmp_dir, 'conala-joined-prod-train.json') 
        valid_file = os.path.join(tmp_dir, 'conala-joined-prod-valid.json')
        
        if tf.gfile.Exists(train_file) or tf.gfile.Exists(valid_file):
            tf.logging.info("Not splitting, file exists")
        else:
            df = self.join_mined_and_train(tmp_dir, extracted_files)
            train, valid = train_test_split(df, test_size=0.10, random_state=42)
            train[['intent_tokens','snippet_tokens']].to_json(train_file)
            valid[['intent_tokens','snippet_tokens']].to_json(valid_file)
        return train_file, valid_file

    def join_mined_and_train(self, tmp_dir, extracted_files):
        df = pd.DataFrame([])
        for extracted_file in extracted_files:
            if 'test' not in extracted_file:
                file_path = os.path.join(tmp_dir, extracted_file)
                df = df.append(pd.read_json(file_path), ignore_index=True, sort=False)
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

        self.maybe_download_conala(tmp_dir)
        extracted_files = extract_data(tmp_dir, False)
        train_filename, valid_filename = self.maybe_split_data(tmp_dir, extracted_files)

        if dataset_split == problem.DatasetSplit.TRAIN:
            df = pd.read_json(train_filename)
            for row in df.itertuples():
                yield {"inputs": " ".join(row.intent_tokens),
                       "targets": " ".join(row.snippet_tokens)}
        elif dataset_split == problem.DatasetSplit.EVAL:
            df = pd.read_json(valid_filename)
            for row in df.itertuples():
                yield {"inputs": " ".join(row.intent_tokens),
                       "targets": " ".join(row.snippet_tokens)}
        else:
            pass
            # TODO: dataset split for test data

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC,
            metrics.Metrics.APPROX_BLEU
        ]

    @classmethod
    def github_data(cls, data_dir, tmp_dir, dataset_split):
        """
        Using data from function_docstring problem
        """
        github = GithubFunctionDocstring()
        return github.generate_samples(data_dir, tmp_dir, dataset_split)

    @classmethod
    def tokenize_code(cls, text: str):
        "A very basic procedure for tokenizing code strings."
        return RegexpTokenizer(r'\w+').tokenize(text)