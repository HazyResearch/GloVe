from __future__ import print_function

import torch
import numpy as np
import time
import os
import struct

import util

def power_iteration(mat, x, x0=None, iterations=50, beta=0.0, norm_freq=1):
    begin = time.time()
    mat = mat.cuda()
    x = x.cuda()
    if beta != 0:
        if x0 is None:
            x0 = torch.zeros([self.n, self.dim]).type(torch.DoubleTensor)
        x0.cuda()
    end = time.time()
    print("GPU Loading:", end - begin)

    for i in range(iterations):
        begin = time.time()
        if beta == 0:
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

    # Scale correctly
    begin = time.time()
    x = torch.mm(mat, x)
    end = time.time()
    print("Final scaling:", end - begin)

    begin = time.time()
    x = x.cpu()
    if beta != 0:
        x0 = x0.cpu()
    end = time.time()
    print("CPU Loading:", end - begin)

    return x, x0

def alecton():
    pass

def vr():
    pass

def sgd():
    pass
