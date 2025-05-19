import random
from itertools import compress
import json
import numpy as np
import torch
import os
from torch.utils.data import Subset


def seed_everything(seed):
    """
    To fix the random seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0, mode=None):
    """
    Split dataset randomly into train, valid, and test subsets.
    
    :param dataset: Dataset object to be split
    :param frac_train: Fraction of the dataset to be used for training
    :param frac_valid: Fraction of the dataset to be used for validation
    :param frac_test: Fraction of the dataset to be used for testing
    :param seed: Random seed for reproducibility
    :return: train, valid, test subsets of the dataset
    """

    # seed_everything(seed)

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Ensure the fractions sum to 1
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    # Get the total number of samples in the dataset
    total_size = len(dataset)

    # Generate indices for the dataset
    indices = np.random.permutation(total_size)

    # Calculate split sizes
    train_size = int(frac_train * total_size)
    valid_size = int(frac_valid * total_size)
    test_size = total_size - train_size - valid_size  # The rest goes to test

    # Split the indices into train, valid, and test sets
    train_idx, valid_idx, test_idx = indices[:train_size], indices[train_size:train_size + valid_size], indices[train_size + valid_size:]

    # Create subsets based on the indices
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    if mode == "index":
        return train_idx, valid_idx, test_idx
    else:
        return train_dataset, valid_dataset, test_dataset



