from __future__ import print_function, absolute_import

import torch
import numpy as np
import time
import os
import struct
import sys

import embedding.util as util

# TODO: automatically match defaults from cmd line?


def power_iteration(mat, x, x0=None, iterations=50, beta=0., norm_freq=1, gpu=False):
    for i in range(iterations):
        begin = time.time()
        if beta == 0.:
            x = util.mm(mat, x, gpu)
        else:
            x, x0 = util.mm(mat, x, gpu) - beta * x0, x
        end = time.time()
        print("Iteration", i + 1, "took", end - begin)
        sys.stdout.flush()

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

    rng = torch.FloatTensor(batch)  # TODO: seems like theres no long random on cuda
    if mat.is_cuda:
        rng = rng.cuda()

    for i in range(iterations):
        begin = time.time()
        # x += eta * torch.mm(torch.spmm(samples, mat), x)
        rng.uniform_(nnz)
        if mat.is_cuda:  # TODO: way to do this without cases?
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

    rng = torch.FloatTensor(batch)  # TODO: seems like theres no long random on cuda
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
            if mat.is_cuda:  # TODO: way to do this without cases?
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


def sgd(mat, x, iterations=50, eta=1e-3, batch=100000):
    # TODO: this does not do any negative sampling
    # TODO: does this need norm_freq

    nnz = mat._nnz()
    n, dim = x.shape

    for i in range(iterations):
        begin = time.time()
        total_cost = 0.
        for start in range(0, nnz, batch):
            end = min(start + batch, nnz)

            X = mat._values()[start:end]

            row = mat._indices()[0, start:end]
            col = mat._indices()[1, start:end]

            pred = (x[row, :] * x[col, :]).sum(1)
            error = pred - torch.log(X)
            step = -eta * error

            dx = step.expand(dim, end - start).t().repeat(2, 1) * x[torch.cat([col, row]), :]
            x.index_add_(0, torch.cat([row, col]), dx)

            total_cost += 0.5 * (error * error).sum()
            print("Iteration", i + 1, "\t", start // batch + 1, "/", (nnz + batch - 1) // batch, "\t", time.time() - begin, end="\r")

        print("Iteration", i + 1, "took", time.time() - begin)
        print("Error:", total_cost / nnz)
        sys.stdout.flush()

    return x
    raise NotImplementedError("SGD solver is not implemented yet.")


def glove(mat, x, bias=None, iterations=50, eta=1e-3, batch=100000):
    # NOTE: this does not include the context vector/bias
    #       the word vector/bias is just used instead

    xmax = 100
    alpha = 0.75

    nnz = mat._nnz()
    n, dim = x.shape

    # TODO: should bias be CPU or GPU
    if bias is None:
        begin = time.time()
        f_mat = mat.clone()
        f_mat._values().div_(xmax).clamp_(max=1)
        # TODO: alpha power

        log_mat = mat.clone()
        log_mat._values().log_()

        log_mat._values().mul_(f_mat._values())
        bias = util.sum_rows(log_mat) / util.sum_rows(f_mat) / 2
        print("Initial bias took", time.time() - begin)

        # bias = torch.cuda.FloatTensor(n)
        # bias.zero_()
        for i in range(100):
            begin = time.time()
            total_cost = 0.
            for start in range(0, nnz, batch):
                end = min(start + batch, nnz)

                X = mat._values()[start:end]

                f = X / xmax
                f.clamp(max=1)
                # TODO alpha power

                row = mat._indices()[0, start:end]
                col = mat._indices()[1, start:end]

                pred = bias[row] + bias[col]
                error = pred - torch.log(X)
                step = -0.001 * f * error

                bias.index_add_(0, torch.cat([row, col]), torch.cat([step, step]))

                total_cost += 0.5 * (f * error * error).sum()
                print("Tune bias ", i + 1, "\t", start // batch + 1, "/", (nnz + batch - 1) // batch, "\t", time.time() - begin, end="\r")
            print("Error:", total_cost / nnz)

    for i in range(iterations):
        begin = time.time()
        total_cost = 0.
        for start in range(0, nnz, batch):
            end = min(start + batch, nnz)

            X = mat._values()[start:end]

            f = X / xmax
            f.clamp(max=1)
            # TODO alpha power

            row = mat._indices()[0, start:end]
            col = mat._indices()[1, start:end]

            pred = (x[row, :] * x[col, :]).sum(1) + bias[row] + bias[col]
            error = pred - torch.log(X)
            step = -eta * f * error

            dx = step.expand(dim, end - start).t().repeat(2, 1) * x[torch.cat([col, row]), :]
            x.index_add_(0, torch.cat([row, col]), dx)
            # bias.index_add_(0, torch.cat([row, col]), torch.cat([step, step]))

            total_cost += 0.5 * (f * error * error).sum()
            print("Iteration", i + 1, "\t", start // batch + 1, "/", (nnz + batch - 1) // batch, "\t", time.time() - begin, end="\r")

        print("Iteration", i + 1, "took", time.time() - begin)
        print("Error:", total_cost / nnz)
        sys.stdout.flush()

    return x, bias
