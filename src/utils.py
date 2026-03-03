# src/utils.py

import random
import numpy as np


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    Controls randomness in Python and NumPy.
    """
    random.seed(seed)
    np.random.seed(seed)