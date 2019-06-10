# Semantic Code Search
Ready-made training and testing scripts are provided in this directory. Below are the instructions to make use of them.

## Overview
For a greater detail of our model, please refer to our [report](report/semantic_code_search.pdf).

All our code is present in the [semantic_search.py](usr_dir/semantic_search.py) file. It makes use of tensor2tensor (T2T) components to work. In fact the file mentioned above serves as an extended component of T2T.

Inside you'll find multiple classes with the `@registry.register_problem` decorator function sitting on top of it. This lets T2T know these are new problems that we want to add to its list of problems. This enables us to use these problems during training and testing by just providing their name, and T2T will know where to find it.

We inherit broad problem classes like `Text2TextProblem` to our semantic search classes and overwrite the relevant functions for it to make it work for our problem. You can refer to the [text_problems.py](tensor2tensor/data_generators/text_problems.py) file to know about the available functions and their purposes.

The 2 most important functions in our code are the `vocab_type` and `generate_samples` functions. The first defines the type of vocab to use (like `SUBWORD`), and the second is responsible for generating training samples. `generate_samples` returns a dictionary with `inputs` and `targets` keys, with each pointing to a single pair of respective parts in the training data. This needs to be `yield`-ed one at a time from the dataset. 

## Installation
Install using `-e` flag with `pip` command by providing the path to the directory holding this repository; more specifically the directory holding `setup.py` file inside the project repository.

If running from inside here:
```bash
pip3 install -e .
```

## Directory Setup
Open the [train_conala.sh](usr_dir/train_conala.sh) and [test_conanal.sh](usr_dir/test_conala.sh) files and change the environment variables defined at the start to point to the right locations. The purpose for each are mentioned below.

* **DATA_DIR** - The location of data files that will be used for generation of samples during training.
* **TRAIN_DIR** - The location where the model checkpoints will be saved.
* **TMP_DIR** - Any data that needs to be downloaded and stored before processing will be stored here.
* **PROBLEM** - The name of the problem class converted from camel case to (lower) snake case.
* **USR_DIR** - The [usr_dir](usr_dir) directory. You can choose to copy this directory elsewhere and it will work just fine. Just update the location accordingly.
* **HPARAMS** - T2T provides with existing hyperparameters presets, so don't need to change this.
* **MODEL** - The type of model to use. Can be left unchanged as well.

Note that all directories need to be created manually, T2T will not create them for you.

## Training and Testing
If all the directories are specified correctly, just run the [train_conala.sh](usr_dir/train_conala.sh) for training, and [test_conanal.sh](usr_dir/test_conala.sh) for testing with the latest model checkpoint.

## Additional Documentation
Additional T2T documentation can be found in the [docs](docs) directory.