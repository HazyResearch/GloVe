from __future__ import print_function

import torch
import numpy as np
import time
import os
import struct

import solver
import util

class Embedding(object):
    def __init__(self, dim=50):
        self.dim = dim

    def load(self, cooccurrence, vocab, embedding=None):
        self.n = cooccurrence.size()[0]

        self.cooccurrence = cooccurrence
        self.cooccurrence._values().log1p_()
        self.vocab = vocab
        # self.words = ["word_" + str(i) for i in range(self.n)]

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

    def solve(self, gpu=True):
        prev = None
        # prev = torch.zeros([self.n, self.dim]).type(torch.DoubleTensor)

        if gpu:
            begin = time.time()
            self.cooccurrence = self.cooccurrence.cuda()
            self.embedding = self.embedding.cuda()
            if prev is not None:
                prev.cuda()
            end = time.time()
            print("GPU Loading:", end - begin)

        self.embedding, _ = solver.power_iteration(self.cooccurrence, self.embedding, x0=prev)
        self.scale(0.5)
        self.normalize_embeddings()

        if gpu:
            begin = time.time()
            self.embedding = self.embedding.cpu()
            end = time.time()
            print("CPU Loading:", end - begin)

    def scale(self, p=1.):
        """Assumes that matrix is normalized."""
        # Scale correctly
        begin = time.time()

        # TODO: faster estimation of eigenvalues?
        temp = torch.mm(self.cooccurrence, self.embedding)
        norm = torch.norm(temp, 2, 0, True)

        norm = norm.pow(p)
        self.embedding = self.embedding.mul(norm.expand_as(self.embedding))
        end = time.time()
        print("Final scaling:", end - begin)

    def normalize_embeddings(self):
        norm = torch.norm(self.embedding, 2, 1, True)
        self.embedding = self.embedding.div(norm.expand_as(self.embedding))

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
    # embedding.load(*util.synthetic(3, 9))
    embedding.solve()
    embedding.save_to_file()
