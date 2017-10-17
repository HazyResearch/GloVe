from __future__ import print_function

import torch
import time
import os

class Embedding(object):
    def __init__(self, dim=300):
        self.dim = dim

    def load(self, cooccurrence, vocab, embedding=None):
        self.n = cooccurrence.size()[0]

        self.cooccurrence = cooccurrence
        self.vocab = vocab

        if embedding is None:
            self.embedding = torch.zeros([self.n, self.dim]).type(torch.DoubleTensor)
            for i in range(min(self.n, self.dim)):
                self.embedding[i, i] = 1
        else:
            self.embedding = embedding

    def load_from_file(self):

        def parse_line(l):
            l = l.split()
            assert(len(l) == 2)
            return l[0], int(l[1])

        with open("vocab.txt") as f:
            lines = [parse_line(l) for l in f]
            self.words = [l[0] for l in lines]
            self.vocab = torch.DoubleTensor([l[1] for l in lines])

        print(os.stat("cooccurrence.shuf.bin"))
        v = torch.randn([nnz])
        v = v.type(torch.DoubleTensor)
        ind = torch.rand(1, nnz) * torch.Tensor([n, n]).repeat(nnz, 1).transpose(0, 1)
        ind = ind.type(torch.LongTensor)
        with open("cooccurrence.shuf.bin") as f:


    def normalize(self):
        begin = time.time()
        self.embedding, r = torch.qr(self.embedding)
        if self.prev is not None:
            self.prev = torch.mm(self.prev, torch.inverse(r))
        end = time.time()
        print("Normalizing took", end - begin)

    def power_iteration(self, iterations=1, beta=0.1, norm_freq=1):
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
        self.embedding = torch.mm(self.cooccurrence, self.embedding)

        begin = time.time()
        self.cooccurrence.cpu()
        self.embedding.cpu()
        end = time.time()
        print("CPU Loading:", end - begin)

    def vr(self):
        pass

    def dump_to_file(self):
        pass

    def get_embeddings(self):
        pass


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
    print("Generating syntetic data:", end - begin)

    return cooccurrence, vocab


def main(argv=None):

    embedding = Embedding()
    embedding.load_from_file()
    embedding.load(*synthetic(500000, 10000000))
    embedding.power_iteration()
    embedding.dump_to_file()
