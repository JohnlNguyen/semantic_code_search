import csv
from pathlib import Path

import pandas as pd
import tensorflow as tf
from six import StringIO

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry


@registry.register_problem
class Conala(text_problems.Text2TextProblem):
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
            "conala-train.json",
            "django-all.anno",
            "django-all.code"
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

        # TODO: Manually separate train/eval data set.
        file_names = self.pair_files_list
        all_files = [
            generator_utils.maybe_download(tmp_dir, file_name, uri)
            for uri, file_name in file_names
        ]

        for file_name in all_files:
            tf.logging.debug("Reading {}".format(file_name))
            if ".jsonl" in file_name:
                contents = Path(file_name).read_text()
                contents = contents.splitlines()
                df = pd.DataFrame([dict(eval(x)) for x in contents])
                for row in df.iterrows():
                    yield Conala.get_row_content(row)
            elif ".json" in file_name:
                df = pd.read_json(file_name)
                for row in df.iterrows():
                    yield Conala.get_row_content(row)
            else:
                # TODO: Figure out how to handle django dataset
                pass

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC,
            metrics.Metrics.APPROX_BLEU
        ]

    @classmethod
    def get_row_content(cls, row):
        if len(row) < 2:
            raise Exception("Row does not have content")
        row = row[1]
        return {"inputs": row.snippet,
                "targets": row.rewritten_intent if 'rewritten_intent' in row and row.rewritten_intent != None else row.intent}
