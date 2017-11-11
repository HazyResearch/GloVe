from __future__ import print_function, absolute_import

import torch
import numpy as np

def sample(mat, batch, scheme="element", random=True):
    n = mat.shape[0]
    nnz = mat._nnz()
    batch = min(batch, nnz)

    if random:
        while True:
            yield
    else:
        while True:
            yield
