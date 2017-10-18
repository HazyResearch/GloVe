from __future__ import print_function

import torch
import numpy as np
import time
import os
import struct

import solver
import util

class Embedding(object):
    def __init__(self, dim=200):
        self.dim = dim

    def load(self, cooccurrence, vocab, embedding=None):
        self.n = cooccurrence.size()[0]

        self.cooccurrence = cooccurrence
        self.cooccurrence._values().log1p_()
        self.vocab = vocab

        if embedding is None:
            self.embedding = torch.randn([self.n, self.dim]).type(torch.DoubleTensor)
            self.embedding, _ = util.normalize(self.embedding)
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
        print("n:", n)

        filesize = os.stat("cooccurrence.shuf.bin").st_size
        assert(filesize % 16 == 0)
        nnz = filesize / 16
        print("nnz:", nnz)
        v = np.empty(nnz, np.float64)
        ind = np.empty((2, nnz), np.int64) # TODO: binary format is int32, but torch uses Long
        with open("cooccurrence.shuf.bin", "rb") as f:
            content = f.read()
            i = 0
            block = 1000
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

    def solve(self):
        self.embedding, _ = solver.power_iteration(self.cooccurrence, self.embedding)

    def save_to_file(self):
        begin = time.time()
        with open("vectors.txt", "w") as f:
            for i in range(self.n):
                f.write(self.words[i] + " " + " ".join([str(self.embedding[i, j]) for j in range(self.dim)]) + "\n")
        end = time.time()
        print("Saving embeddings:", end - begin)


def main(argv=None):
    embedding = Embedding()
    embedding.load_from_file()
    # embedding.load(*util.synthetic(500, 10000))
    embedding.solve()
    embedding.save_to_file()
