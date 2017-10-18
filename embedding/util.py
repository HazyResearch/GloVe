from __future__ import print_function

import torch
import time

def synthetic(n, nnz):
    """This function generates a synthetic matrix."""
    begin = time.time()
    v = torch.randn([nnz])
    v = v.type(torch.DoubleTensor)
    ind = torch.rand(1, nnz) * torch.Tensor([n, n]).repeat(nnz, 1).transpose(0, 1)
    ind = ind.type(torch.LongTensor)

    cooccurrence = torch.sparse.DoubleTensor(ind, v, torch.Size([n, n])).coalesce()
    vocab = None
    end = time.time()
    print("Generating synthetic data:", end - begin)

    return cooccurrence, vocab

def normalize(x, x0=None):
    # TODO: is it necessary to reorder columns by magnitude
    begin = time.time()
    norm = torch.norm(x, 2, 0, True)
    print(norm)
    # x = x.div(norm.expand_as(x))
    x, r = torch.qr(x)
    norm = torch.norm(x, 2, 0, True)
    print(norm)
    if x0 is not None:
        x0 = torch.mm(x0, torch.inverse(r))
    end = time.time()
    print("Normalizing took", end - begin)

    return x, x0
