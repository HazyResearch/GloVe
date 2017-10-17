from __future__ import print_function

import torch
import numpy as np
import time
import os
import struct

class Embedding(object):
    def __init__(self, dim=300):
        self.dim = dim

    def load(self, cooccurrence, vocab, embedding=None):
        self.n = cooccurrence.size()[0]

        self.cooccurrence = cooccurrence
        self.vocab = vocab
        self.prev = None

        if embedding is None:
            self.embedding = torch.randn([self.n, self.dim]).type(torch.DoubleTensor)
            self.normalize()
        else:
            self.embedding = embedding

    def load_from_file(self):

        begin = time.time()

        def parse_line(l):
            l = l.split()
            assert(len(l) == 2)
            return l[0], int(l[1])

        with open("vocab.txt") as f:
            lines = [parse_line(l) for l in f]
            self.words = [l[0] for l in lines]
            vocab = torch.DoubleTensor([l[1] for l in lines])
        n = vocab.size()[0]

        filesize = os.stat("cooccurrence.shuf.bin").st_size
        assert(filesize % 16 == 0)
        nnz = filesize / 16
        # v = np.zeros(nnz, np.float64)
        # v = torch.zeros([nnz])
        # v = v.type(torch.DoubleTensor)
        # ind = torch.zeros(2, nnz)
        # ind = ind.type(torch.LongTensor)
        v = np.empty(nnz, np.float64)
        ind = np.empty((2, nnz), np.int64) # TODO: binary format is int32, but torch uses Long
        with open("cooccurrence.shuf.bin", "rb") as f:
            content = f.read()
            # for i in range(min(nnz, 10000000)):
            # for i in range(min(nnz, 100000)):
            i = 0
            block = 100
            while i < nnz:
                block = min(block, nnz - i)
                line = struct.unpack("iid" * block, content[(16 * i):(16 * (i + block))])
                ind[0, i:(i + block)] = line[0::3]
                ind[1, i:(i + block)] = line[1::3]
                v[i:(i + block)] = line[2::3]
                i += block
            ind = ind - 1
        v = torch.DoubleTensor(v)
        ind = torch.LongTensor(ind)
        cooccurrence = torch.sparse.DoubleTensor(ind, v, torch.Size([n, n])).coalesce()

        self.load(cooccurrence, vocab)

        end = time.time()
        print("Loading data took", end - begin)

    def normalize(self):
        # TODO: is it necessary to reorder columns by magnitude
        begin = time.time()
        self.embedding, r = torch.qr(self.embedding)
        if self.prev is not None:
            self.prev = torch.mm(self.prev, torch.inverse(r))
        end = time.time()
        print("Normalizing took", end - begin)

    def power_iteration(self, iterations=1, beta=0.0, norm_freq=1):
        begin = time.time()
        self.cooccurrence.cuda()
        self.embedding.cuda()
        if beta == 0:
            self.prev = None
        else:
            self.prev = torch.zeros([self.n, self.dim]).type(torch.DoubleTensor)
            self.prev.cuda()
        end = time.time()
        print("GPU Loading:", end - begin)

        for i in range(iterations):
            begin = time.time()
            if beta == 0:
                self.embedding = torch.mm(self.cooccurrence, self.embedding)
            else:
                self.embedding = torch.mm(self.cooccurrence, self.embedding) - beta * self.prev
            end = time.time()
            print("Iteration", i + 1, "took", end - begin)

            if (i + 1) % norm_freq == 0:
                self.normalize()

        if iterations % norm_freq != 0:
            # Only normalize if the last iteration did not
            self.normalize()

        # Scale correctly
        begin = time.time()
        self.embedding = torch.mm(self.cooccurrence, self.embedding)
        end = time.time()
        print("Final scaling:", end - begin)

        begin = time.time()
        self.embedding.cpu()
        end = time.time()
        print("CPU Loading:", end - begin)

    def vr(self):
        pass

    def save_to_file(self):
        begin = time.time()
        with open("vectors.txt", "w") as f:
            for i in range(self.n):
                f.write(self.words[i] + " " + " ".join([str(self.embedding[i, j]) for j in range(self.dim)]) + "\n")
        end = time.time()
        print("Saving embeddings:", end - begin)


def synthetic(n, nnz):
    """This function generates a synthetic matrix."""
    begin = time.time()
    v = torch.randn([nnz])
    v = v.type(torch.DoubleTensor)
    ind = torch.rand(1, nnz) * torch.Tensor([n, n]).repeat(nnz, 1).transpose(0, 1)
    ind = ind.type(torch.LongTensor)
    # for i in range(nnz):
    #     ind[0, i] = i
    #     ind[1, i] = i
    #     v[i] = 2

    cooccurrence = torch.sparse.DoubleTensor(ind, v, torch.Size([n, n])).coalesce()
    vocab = None
    end = time.time()
    print("Generating synthetic data:", end - begin)

    return cooccurrence, vocab


def main(argv=None):

    embedding = Embedding()
    embedding.load_from_file()
    # embedding.load(*synthetic(500000, 10000000))
    embedding.power_iteration()
    embedding.save_to_file()
