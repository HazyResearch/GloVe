from __future__ import print_function, absolute_import

import torch
import numpy as np

def sample(n, nnz, batch, scheme="element", random=True):
    batch = min(nnz, batch)

    while True:
        yield
