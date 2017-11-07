from __future__ import print_function, absolute_import

import torch
import numba
import numpy as np
import time
import sys
import argparse
import logging
import scipy

import embedding.tensor_type as tensor_type


def synthetic(n, nnz):
    """This function generates a synthetic matrix."""
    begin = time.time()
    # TODO: distribute as power law?
    #       (closer to real distribution)
    v = torch.abs(torch.randn([nnz]))
    # TODO: make non-neg
    v = v.type(torch.DoubleTensor)
    ind = torch.rand(2, nnz) * torch.Tensor([n, n]).repeat(nnz, 1).transpose(0, 1)
    # TODO: fix ind (only diag right now)
    ind = ind.type(torch.LongTensor)

    cooccurrence = torch.sparse.DoubleTensor(ind, v, torch.Size([n, n])).coalesce()
    vocab = None
    words = None
    logger = logging.getLogger(__name__)
    logger.info("Generating synthetic data: " + str(time.time() - begin))

    return cooccurrence, vocab, words


def normalize(x, x0=None):

    logger = logging.getLogger(__name__)

    # TODO: is it necessary to reorder columns by magnitude
    # TODO: more numerically stable implementation?
    begin = time.time()
    norm = torch.norm(x, 2, 0, True).squeeze()
    logger.info(" ".join(["{:10.2f}".format(n) for n in norm]))
    a = time.time()
    _, perm = torch.sort(-norm)
    norm = norm[perm]
    x = x[:, perm]
    if x0 is not None:
        x0 = x0[:, perm]
    logger.info("Permute time: " + str(time.time() - a))
    try:
        temp, r = torch.qr(x)
    except RuntimeError as e:
        logger.error("QR decomposition has run into a problem\n"
                     "Older versions of pytoch had a memory leak in QR:\n"
                     "    https://github.com/pytorch/pytorch/issues/3009\n"
                     "Updating pytorch may fix this issue.")
        raise e
    if np.isnan(torch.sum(temp)):
        # qr seems to occassionally be unstable and result in nan
        logger.warn("QR decomposition resulted in NaNs\n"
                    "Normalizing, but not orthogonalizing")
        # TODO: should a little bit of jitter be added to make qr succeed?
        x = x.div(norm.expand_as(x))
        if x0 is not None:
            x0 = x0.div(norm.expand_as(x0))
    else:
        x = temp
        if x0 is not None:
            x0 = torch.mm(x0, torch.inverse(r))
    logger.info("Normalizing took " + str(time.time() - begin))

    return x, x0


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mm(A, x, gpu=False):

    logger = logging.getLogger(__name__)

    if (type(A) == scipy.sparse.csr.csr_matrix or
        type(A) == scipy.sparse.coo.coo_matrix or
        type(A) == scipy.sparse.csc.csc_matrix):
        return torch.from_numpy(A * x.numpy())
    elif not (A.is_cuda or x.is_cuda or gpu):
        # Data and computation on CPU
        return torch.mm(A, x)
    else:
        # Compute on GPU, regardless of where data is
        if A.is_cuda and x.is_cuda:
            # Everything on GPU anyways, just multiply normally
            # TODO: workaround for pytorch memory leak
            return torch.mm(A, x)
        else:

            if (A.type() == "torch.sparse.FloatTensor" or
                A.type() == "torch.cuda.sparse.FloatTensor"):
                SparseTensor = torch.cuda.sparse.FloatTensor
            elif (A.type() == "torch.sparse.DoubleTensor" or
                  A.type() == "torch.cuda.sparse.DoubleTensor"):
                SparseTensor = torch.cuda.sparse.DoubleTensor
            else:
                raise NotImplementedError("Type of cooccurrence matrix (" + A.type() + ") is not recognized.")

            n, dim = x.shape
            nnz = A._nnz()

            indices = A._indices().t()
            values = A._values()

            # TODO: automate batch choice
            A_batches = 35
            x_batches = 5
            if A.is_cuda:
                A_batches = 1
            if x.is_cuda:
                x_batches = 1
            newx = 0 * x
            for i in range(A_batches):
                if A.is_cuda:
                    sample = A
                else:
                    start = i * nnz // A_batches
                    end = (i + 1) * nnz // A_batches

                    ind = indices[start:end, :]
                    val = values[start:end]

                    # TODO: resort to sync transfer if needed
                    try:
                        ind = ind.cuda(async=True)
                        val = val.cuda(async=True)
                    except RuntimeError as e:
                        # logging.warn("async transfer failed")
                        ind = ind.cuda()
                        val = val.cuda()

                    sample = SparseTensor(ind.t(), val, torch.Size([n, n]))

                for j in range(x_batches):
                    logging.info(str(i) + " / " + str(A_batches) + "\t" + str(j) + " / " + str(x_batches) + "\r")
                    sys.stdout.flush()

                    if x.is_cuda:
                        newx = newx.addmm(sample, x)
                    else:
                        start = j * dim // x_batches
                        end = (j + 1) * dim // x_batches

                        cols = x[:, start:end]

                        try:
                            cols = cols.cuda(async=True)
                        except RuntimeError as e:
                            # logging.warn("async transfer failed")
                            cols = cols.cuda()

                        cols = torch.mm(sample, cols).cpu()
                        newx[:, start:end] += cols

            return newx


def sum_rows(A):
    n = A.shape[0]
    if A.is_cuda:
        ones = tensor_type.to_dense(A.type())(n, 1)
        ones.fill_(1)
        return torch.mm(A, ones).squeeze(1)
    else:
        @numba.jit(nopython=True, cache=True)
        def sr(n, ind, val):
            nnz = val.shape[0]
            ans = np.zeros(n, dtype=val.dtype)
            for i in range(nnz):
                ans[ind[0, i]] += val[i]
            return ans
        return tensor_type.to_dense(A.type())(sr(A.shape[0], A._indices().numpy(), A._values().numpy()))
        # return torch.from_numpy(scipy.sparse.coo_matrix((A._values().numpy(), (A._indices()[0, :].numpy(), A._indices()[1, :].numpy())), shape=A.shape).sum(1)).squeeze()
