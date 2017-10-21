from __future__ import print_function

import torch
import numpy as np
import time
import os
import struct

import util

def power_iteration(mat, x, x0=None, iterations=50, beta=0., norm_freq=1):
    for i in range(iterations):
        begin = time.time()
        if beta == 0.:
            x = torch.mm(mat, x)
        else:
            x, x0 = torch.mm(mat, x) - beta * x0, x
        end = time.time()
        print("Iteration", i + 1, "took", end - begin)

        if (i + 1) % norm_freq == 0:
            x, x0 = util.normalize(x, x0)

    if iterations % norm_freq != 0:
        # Only normalize if the last iteration did not
        x, x0 = util.normalize(x, x0)

    return x, x0

def alecton():
    pass

def vr():
    pass

def sgd():
    pass
