from __future__ import print_function

import torch
import numpy as np
import time
import os
import struct

import util

# TODO: automatically match defaults from cmd line?

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

def alecton(mat, x, iterations=50, eta=1e-3, norm_freq=1, batch=100000):
    # TODO: alecton will need a lot more iterations (since one iteration does
    #       much less work) -- clean way to have different defaults?
    n = mat.shape[0]
    nnz, = mat._values().shape
    batch = min(batch, nnz)

    # ind = torch.LongTensor(np.random.choice(n, [1, batch], False).repeat(2, 0))
    # v = torch.DoubleTensor(batch).fill_(n / float(batch))
    # samples = torch.sparse.DoubleTensor(ind, v, torch.Size([n, n]))
    # if mat.is_cuda:
    #     samples = samples.cuda()

    rng = torch.FloatTensor(batch) # TODO: seems like theres no long random on cuda
    if mat.is_cuda:
        rng = rng.cuda()

    for i in range(iterations):
        begin = time.time()
        # x += eta * torch.mm(torch.spmm(samples, mat), x)
        rng.uniform_(nnz)
        if mat.is_cuda: # TODO: way to do this without cases?
            elements = rng.type(torch.cuda.LongTensor)
        else:
            elements = rng.type(torch.LongTensor)
        ind = mat._indices()[:, elements]
        v = mat._values()[elements]
        if mat.is_cuda:
            sample = torch.cuda.sparse.DoubleTensor(ind, v, torch.Size([n, n]))
        else:
            sample = torch.sparse.DoubleTensor(ind, v, torch.Size([n, n]))
        sample = nnz / float(batch) * sample

        x += eta * torch.mm(sample, x)
        end = time.time()
        print("Iteration", i + 1, "took", end - begin)

        if (i + 1) % norm_freq == 0:
            x, _ = util.normalize(x, None)

    if iterations % norm_freq != 0:
        # Only normalize if the last iteration did not
        x, _ = util.normalize(x, None)

    return x

def vr(mat, x, x0=None, iterations=50, beta=0., norm_freq=1, batch=100000, innerloop=10):
    n = mat.shape[0]
    nnz, = mat._values().shape
    batch = min(batch, nnz)

    rng = torch.FloatTensor(batch) # TODO: seems like theres no long random on cuda
    if mat.is_cuda:
        rng = rng.cuda()

    for i in range(iterations):
        begin = time.time()
        xtilde = x.clone()
        gx = torch.mm(mat, xtilde)
        for j in range(innerloop):
            # TODO: can ang be generated without expand_as?
            ang = torch.sum(x * xtilde, 0).expand_as(xtilde)

            rng.uniform_(nnz)
            if mat.is_cuda: # TODO: way to do this without cases?
                elements = rng.type(torch.cuda.LongTensor)
            else:
                elements = rng.type(torch.LongTensor)
            ind = mat._indices()[:, elements]
            v = mat._values()[elements]
            if mat.is_cuda:
                sample = torch.cuda.sparse.DoubleTensor(ind, v, torch.Size([n, n]))
            else:
                sample = torch.sparse.DoubleTensor(ind, v, torch.Size([n, n]))
            sample = nnz / float(batch) * sample

            if beta == 0:
                x = torch.mm(sample, x) - ang * torch.mm(sample, xtilde) + ang * gx
            else:
                x, x0 = torch.mm(sample, x) - ang * torch.mm(sample, xtilde) + ang * gx - beta * x0, x

            # TODO: option to normalize in inner loop

        end = time.time()
        print("Iteration", i + 1, "took", end - begin)

        if (i + 1) % norm_freq == 0:
            x, x0 = util.normalize(x, x0)

    if iterations % norm_freq != 0:
        # Only normalize if the last iteration did not
        x, x0 = util.normalize(x, x0)

    return x, x0

def sgd(mat, x, iterations=50, eta=1e-3, norm_freq=1, batch=100000):
    raise NotImplementedError("SGD solver is not implemented yet.")
