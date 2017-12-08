from __future__ import print_function, absolute_import

import torch
import numpy as np
import time
import os
import struct
import sys
import sparsesvd
import scipy.sparse
import logging
import math

import embedding.util as util

# TODO: automatically match defaults from cmd line?


def power_iteration(mat, x, x0=None, iterations=50, beta=0., norm_freq=1, gpu=False, checkpoint=lambda x, i: None):

    logger = logging.getLogger(__name__)

    for i in range(iterations):
        begin = time.time()
        if beta == 0.:
            x = util.mm(mat, x, gpu)
        else:
            x, x0 = util.mm(mat, x, gpu) - beta * x0, x
        logging.info("Iteration " + str(i + 1) + " took " + str(time.time() - begin))

        if ((i + 1) % norm_freq == 0 or
            (i + 1) == iterations):
            x, x0 = util.normalize(x, x0)

        checkpoint(x, i)

    return x, x0


def alecton(mat, x, iterations=50, eta=1e-3, norm_freq=1, sample=None, gpu=False, checkpoint=lambda x, i: None):

    logger = logging.getLogger(__name__)

    if sample is None:
        sample = util.get_sampler(mat, 100000)

    # TODO: alecton will need a lot more iterations (since one iteration does
    #       much less work) -- clean way to have different defaults?
    n = mat.shape[0]
    nnz = mat._nnz()

    for i in range(iterations):
        begin = time.time()

        m = next(sample)

        x = (1 - eta) * x + eta * util.mm(m, x, gpu)
        logging.info("Iteration " + str(i + 1) + " took " + str(time.time() - begin))

        if ((i + 1) % norm_freq == 0 or
            (i + 1) == iterations):
            x, _ = util.normalize(x, None)

        checkpoint(x, i)

    return x


def vr(mat, x, x0=None, iterations=50, eta=1e-3, beta=0., norm_freq=1, innerloop=10, sample=None, gpu=False, checkpoint=lambda x, i: None):

    logger = logging.getLogger(__name__)

    n, dim = x.shape

    if sample is None:
        sample = util.get_sampler(mat, 100000)

    for i in range(iterations):
        begin = time.time()
        xtilde = x.clone()
        xtilde_norm = torch.sum(xtilde * xtilde, 0)
        gx = torch.mm(mat, xtilde)
        cumprod = type(x)(1, dim)
        cumprod.fill_(1)
        for j in range(innerloop):
            # TODO: can ang be generated without expand_as?
            x_norm = torch.sum(x * x, 0)
            x_xtilde = torch.sum(x * xtilde, 0)
            ang = (x_xtilde / x_norm / xtilde_norm).expand_as(xtilde)

            m = next(sample)

            if beta == 0:
                x = (1 - eta) * x + eta * (util.mm(m, x) - ang * util.mm(m, xtilde) + ang * gx)
            else:
                x, x0 = (1 - eta) * x + eta * (util.mm(m, x) - ang * util.mm(m, xtilde) + ang * gx - beta * x0), (1 - eta) * x0 + eta * x

            n = x.norm(2, 0, True)
            x /= n.expand_as(x)
            if x0 is not None:
                x0 /= n.expand_as(x0)
            cumprod *= n / n[0, 0]
            # TODO: option to normalize in inner loop

        logger.info("Iteration " + str(i + 1) + " took " + str(time.time() - begin))

        x *= cumprod.expand_as(x)
        if x0 is not None:
            x0 *= cumprod.expand_as(x0)
        if ((i + 1) % norm_freq == 0 or
            (i + 1) == iterations):
            x, x0 = util.normalize(x, x0)

        checkpoint(x, i)

    return x, x0


def sgd(mat, x, iterations=50, eta=1e-3, sample=None, gpu=False, checkpoint=lambda x, i: None):
    # TODO: this does not do any negative sampling
    # TODO: does this need norm_freq

    nnz = mat._nnz()
    n, dim = x.shape

    for i in range(iterations):
        begin = time.time()

        m = next(sample)

        X = m._values()

        row = m._indices()[0, :]
        col = m._indices()[1, :]

        pred = (x[row, :] * x[col, :]).sum(1)
        error = pred - X
        step = -eta * error / m._values().shape[0]

        dx = step.expand(dim, m._values().shape[0]).t().repeat(2, 1) * x[torch.cat([col, row]), :]
        x.index_add_(0, torch.cat([row, col]), dx)

        logging.info("Iteration " + str(i + 1) + " took " + str(time.time() - begin))
        logging.info("Error: " + str(torch.abs(error).sum() / m._values().shape[0]))

        checkpoint(x, i)

    return x


def poincare(mat, x, iterations=50, eta=1e-3, sample=None, gpu=False, checkpoint=lambda x, i: None, hyperbolic=True):
    # TODO: this does not do any negative sampling

    nnz = mat._nnz()
    n, dim = x.shape

    r = 5.
    t = 1.

    for i in range(iterations):
        begin = time.time()

        m = next(sample)

        X = m._values()
        X = (X != 0).type(type(X))

        row = m._indices()[0, :]
        col = m._indices()[1, :]

        u = x[row, :]
        v = x[col, :]

        d = (u - v)
        if hyperbolic:
            d = 1 + 2 * torch.sum(d * d, 1) / (1 - torch.sum(u * u, 1)) / (1 - torch.sum(v * v, 1))
            d = torch.log(d + torch.sqrt((d - 1) * (d + 1))) # arccosh
        else:
            d = torch.sqrt(torch.sum(d * d, 1))

        dlogpdd = -(2 * X - 1) * torch.exp(d / t) / (t * math.exp(r / t) + t * torch.exp(d / t))

        dddu = 2 * (u - v)
        dddv = 2 * (v - u)
        if hyperbolic:
            dddu = torch.pow((1 - torch.sum(u * u, 1, True)), 2).expand_as(dddu) / 4 * dddu
            dddv = torch.pow((1 - torch.sum(v * v, 1, True)), 2).expand_as(dddv) / 4 * dddv
        
        dx = torch.cat([dlogpdd.view(-1, 1).expand_as(dddu) * dddu, dlogpdd.view(-1, 1).expand_as(dddv) * dddv])
        x.index_add_(0, torch.cat([row, col]), eta * dx)

        if hyperbolic:
            eps = 1e-5
            norm = torch.sum(x[torch.cat([row, col]), :] * x[torch.cat([row, col]), :], 1, True) + eps
            norm.clamp_(min=1)
            x[torch.cat([row, col]), :] /= norm.expand_as(x[torch.cat([row, col]), :])

        logging.info("Iteration " + str(i + 1) + " took " + str(time.time() - begin))
        p = (1 - X) + (2 * X - 1) * 1. / (torch.exp((d - r) / t) + 1)
        error = torch.log(p)
        logging.info("Loss: " + str(error.sum() / m._values().shape[0]))

        u = x[row, :]
        v = x[col, :]

        d = (u - v)
        if hyperbolic:
            d = 1 + 2 * torch.sum(d * d, 1) / (1 - torch.sum(u * u, 1)) / (1 - torch.sum(v * v, 1))
            d = torch.log(d + torch.sqrt((d - 1) * (d + 1))) # arccosh
        else:
            d = torch.sqrt(torch.sum(d * d, 1))
        p = (1 - X) + (2 * X - 1) * 1. / (torch.exp((d - r) / t) + 1)
        error = torch.log(p)
        logging.info("Loss: " + str(error.sum() / m._values().shape[0]))

        checkpoint(x, i)

    return x


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
        f_mat._values().div_(xmax).clamp_(max=1).pow_(alpha)

        log_mat = mat.clone()
        log_mat._values().log_()

        log_mat._values().mul_(f_mat._values())
        bias = util.sum_rows(log_mat) / util.sum_rows(f_mat) / 2
        logging.info("Initial bias took" + str(time.time() - begin))

        # bias = torch.cuda.FloatTensor(n)
        # bias.zero_()
        for i in range(100):
            begin = time.time()
            total_cost = 0.
            for start in range(0, nnz, batch):
                end = min(start + batch, nnz)

                X = mat._values()[start:end]

                f = X / xmax
                f.clamp_(max=1)
                f.pow_(alpha)

                row = mat._indices()[0, start:end]
                col = mat._indices()[1, start:end]

                pred = bias[row] + bias[col]
                error = pred - torch.log(X)
                step = -0.001 * f * error

                bias.index_add_(0, torch.cat([row, col]), torch.cat([step, step]))

                total_cost += 0.5 * (f * error * error).sum()
                logging.info("Tune bias " + str(i + 1) + "\t" + str(start // batch + 1) + " / " + str((nnz + batch - 1) // batch) + "\t" + str(time.time() - begin) + "\r")
            logging.info("Error: " + str(total_cost / nnz))

    for i in range(iterations):
        begin = time.time()
        total_cost = 0.
        for start in range(0, nnz, batch):
            end = min(start + batch, nnz)

            X = mat._values()[start:end]

            f = X / xmax
            f.clamp_(max=1)
            f.pow_(alpha)

            row = mat._indices()[0, start:end]
            col = mat._indices()[1, start:end]

            pred = (x[row, :] * x[col, :]).sum(1) + bias[row] + bias[col]
            error = pred - torch.log(X)
            step = -eta * f * error

            dx = step.expand(dim, end - start).t().repeat(2, 1) * x[torch.cat([col, row]), :]
            x.index_add_(0, torch.cat([row, col]), dx)
            # bias.index_add_(0, torch.cat([row, col]), torch.cat([step, step]))

            total_cost += 0.5 * (f * error * error).sum()
            logging.info("Iteration " + str(i + 1) + "\t" + str(start // batch + 1) + " / " + str((nnz + batch - 1) // batch) + "\t" + str(time.time() - begin) + "\r")

        logging.info("Iteration " + str(i + 1) + " took " + str(time.time() - begin))
        logging.info("Error: " + str(total_cost / nnz))

    return x, bias


def sparseSVD(mat, dim):
    begin = time.time()
    mat = mat.tocsc()
    logging.info("CSC conversion took " + str(time.time() - begin))

    begin = time.time()
    u, s, v = sparsesvd.sparsesvd(mat, dim)
    logging.info("Solving took " + str(time.time() - begin))

    return torch.from_numpy(u.transpose())
