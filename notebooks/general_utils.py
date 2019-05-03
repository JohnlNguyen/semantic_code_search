from pathlib import Path
import logging
import wget
import pickle
import pandas as pd
import numpy as np
from typing import List, Callable, Union, Any, Dict
from more_itertools import chunked
from itertools import chain
import nmslib
from pathos.multiprocessing import Pool, cpu_count
from math import ceil
from collections import Counter

class TokenList:
    def __init__(self, token_list):
        self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
        self.t2id = {v: k for k, v in enumerate(self.id2t)}

    def id(self, x):
        return self.t2id.get(x, 1)

    def token(self, x):
        return self.id2t[x]

    @property
    def length(self):
        return len(self.id2t)

    @property
    def start_id(self):
        return 2

    @property
    def end_id(self):
        return 3

    def id(self, x): return self.t2id.get(x, 1)

    def token(self, x): return self.id2t[x]

    def num(self): return len(self.id2t)

    def startid(self): return 2

    def endid(self): return 3
    
def pad_to_longest(xs, tokens, max_len=999):
    longest = min(len(max(xs, key=len))+2, max_len)
    X = np.zeros((len(xs), longest), dtype='int32')
    X[:,0] = tokens.start_id
    for i, x in enumerate(xs):
        x = x[:max_len-2]
        for j, z in enumerate(x):
            X[i,1+j] = tokens.id(z)
        X[i,1+len(x)] = tokens.end_id
    return X

def create_token_map(data: List[str], delimiter=' ') -> Dict:
    return Counter(create_word_list(data))

def create_word_list(data: List[str], delimiter=' '):
    return flattenlist([w.split(delimiter) for w in data])

def build_vocab(word_dict: Dict, min_freq=5, outpath=None) -> TokenList:
    # filter out word less than min_freq
    word_list = list(list(zip(*filter(lambda wc: wc[1] >= min_freq, word_dict.items())))[0])
    # save to file
    if outpath != None:
        pd.Series(word_list).to_csv(outpath, sep='\n')
    return TokenList(word_list)

def build_data(source_data, target_data, src_tokens, tar_tokens, delimiter=' ', h5_file=None, max_len=200):
    if h5_file is not None and os.path.exists(h5_file):
        print('loading', h5_file)
        with h5py.File(h5_file) as dfile:
            X, Y = dfile['X'][:], dfile['Y'][:]
        return X, Y
    
    source = create_word_list(source_data)
    target = create_word_list(target_data)
    
    X, Y = pad_to_longest(source, src_tokens, max_len), pad_to_longest(target, tar_tokens, max_len)
    if h5_file is not None:
        with h5py.File(h5_file, 'w') as dfile:
            dfile.create_dataset('X', data=X)
            dfile.create_dataset('Y', data=Y)
    return X, Y

def save_file_pickle(fname:str, obj:Any):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_file_pickle(fname:str):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
        return obj


def read_training_files(data_path:str):
    """
    Read data from directory
    """
    PATH = Path(data_path)

    with open(PATH/'train.function', 'r') as f:
        t_enc = f.readlines()

    with open(PATH/'valid.function', 'r') as f:
        v_enc = f.readlines()

    # combine train and validation and let keras split it randomly for you
    tv_enc = t_enc + v_enc

    with open(PATH/'test.function', 'r') as f:
        h_enc = f.readlines()

    with open(PATH/'train.docstring', 'r') as f:
        t_dec = f.readlines()

    with open(PATH/'valid.docstring', 'r') as f:
        v_dec = f.readlines()

    # combine train and validation and let keras split it randomly for you
    tv_dec = t_dec + v_dec

    with open(PATH/'test.docstring', 'r') as f:
        h_dec = f.readlines()

    logging.warning(f'Num rows for encoder training + validation input: {len(tv_enc):,}')
    logging.warning(f'Num rows for encoder holdout input: {len(h_enc):,}')

    logging.warning(f'Num rows for decoder training + validation input: {len(tv_dec):,}')
    logging.warning(f'Num rows for decoder holdout input: {len(h_dec):,}')

    return tv_enc, h_enc, tv_dec, h_dec


def apply_parallel(func: Callable,
                   data: List[Any],
                   cpu_cores: int = None) -> List[Any]:
    """
    Apply function to list of elements.
    Automatically determines the chunk size.
    """
    if not cpu_cores:
        cpu_cores = cpu_count()

    try:
        chunk_size = ceil(len(data) / cpu_cores)
        pool = Pool(cpu_cores)
        transformed_data = pool.map(func, chunked(data, chunk_size), chunksize=1)
    finally:
        pool.close()
        pool.join()
        return transformed_data


def flattenlist(listoflists:List[List[Any]]):
    return list(chain.from_iterable(listoflists))


processed_data_filenames = [
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test.docstring',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test_original_function.json.gz',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train.docstring',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train_original_function.json.gz',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.docstring',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid_original_function.json.gz',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings_original_function.json.gz']


def get_step2_prerequisite_files(output_directory):
    outpath = Path(output_directory)
    assert not list(outpath.glob('*')), f'There are files in {str(outpath.absolute())}, please clear files or specify an empty folder.'
    outpath.mkdir(exist_ok=True)
    print(f'Saving files to {str(outpath.absolute())}')
    for url in processed_data_filenames:
        print(f'downloading {url}')
        wget.download(url, out=str(outpath.absolute()))


def create_nmslib_search_index(numpy_vectors):
    """Create search index using nmslib.

    Parameters
    ==========
    numpy_vectors : numpy.array
        The matrix of vectors

    Returns
    =======
    nmslib object that has index of numpy_vectors
    """

    search_index = nmslib.init(method='hnsw', space='cosinesimil')
    search_index.addDataPointBatch(numpy_vectors)
    search_index.createIndex({'post': 2}, print_progress=True)
    return search_index
