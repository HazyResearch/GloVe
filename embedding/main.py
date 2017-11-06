from __future__ import print_function, absolute_import

import torch
import numpy as np
import time
import os
import struct
import argparse
import sys
import subprocess
import math
import logging
import pandas
import collections

import embedding.solver as solver
import embedding.util as util
import embedding.evaluate as evaluate
import embedding.tensor_type as tensor_type
import embedding.parser as parser
import embedding.logging_config as logging_config


def main(argv=None):

    # Parse command line arguments
    args = parser.get_parser().parse_args(argv)

    # Set up logging for package
    logging_config.init(args.logging)
    logger = logging.getLogger(__name__)

    if args.task == "cooccurrence":
        subprocess.call([os.path.join(os.path.dirname(__file__), "..", "cooccurrence.sh"), args.text])
    elif args.task == "compute":
        if args.gpu and not torch.cuda.is_available():
            logger.warn("GPU use requested, but GPU not available. "
                        "Toggling off GPU use.")
            args.gpu = False
            args.matgpu = False
            args.embedgpu = False

        if args.gpu and (args.solver == "sparsesvd" or args.solver == "gensim"):
            logger.warn("SparseSVD and gensim are not implemented for GPU. "
                        "Toggling off GPU use.")
            args.gpu = False
            args.matgpu = False
            args.embedgpu = False

        if args.solver == "glove" and args.preprocessing != "none":
            logger.warn("GloVe only behaves properly with no preprocessing. "
                        "Turning off preprocessing.")
            args.preprocessing = "none"

        CpuTensor = torch.FloatTensor
        if args.precision == "float":
            CpuTensor = torch.FloatTensor
        elif args.precision == "double":
            CpuTensor = torch.DoubleTensor
        else:
            logger.warn("Precision \"" + args.precision + "\" is not recognized. "
                        "Defaulting to \"float\".")

        embedding = Embedding(args.dim, args.gpu, args.matgpu, args.embedgpu, CpuTensor)
        embedding.load_from_file(args.vocab, args.cooccurrence, args.initial, args.initialbias)
        # embedding.load(*util.synthetic(2, 4))
        embedding.preprocessing(args.preprocessing)
        embedding.solve(mode=args.solver, gpu=args.gpu, scale=args.scale, normalize=args.normalize, iterations=args.iterations, eta=args.eta, momentum=args.momentum, normfreq=args.normfreq, batch=args.batch, innerloop=args.innerloop)
        embedding.save_to_file(args.vectors)
    elif args.task == "evaluate":
        evaluate.evaluate(args.vocab, args.vectors)


class Embedding(object):
    def __init__(self, dim=50, gpu=True, matgpu=None, embedgpu=None, CpuTensor=torch.FloatTensor):
        self.dim = dim
        self.gpu = gpu

        # TODO: add warning for storage on gpu when computation is on gpu
        # TODO: swap off storage if too much memory
        if matgpu is None:
            matgpu = gpu
        if embedgpu is None:
            embedgpu = gpu
        self.matgpu = matgpu
        self.embedgpu = embedgpu

        self.CpuTensor = CpuTensor

        self.logger = logging.getLogger(__name__)

    def load(self, cooccurrence, vocab, words, embedding=None):
        self.n = cooccurrence.size()[0]

        # TODO: error if n > dim

        self.cooccurrence = cooccurrence
        self.vocab = vocab
        self.words = words
        self.embedding = embedding

    def load_from_file(self, vocab_file="vocab.txt", cooccurrence_file="cooccurrence.shuf.bin", initial_vectors=None, initial_bias=None):
        begin = time.time()

        def parse_line(l):
            l = l.split()
            assert(len(l) == 2)
            return l[0], int(l[1])

        with open(vocab_file) as f:
            lines = [parse_line(l) for l in f]
            words = [l[0] for l in lines]
            vocab = self.CpuTensor([l[1] for l in lines])
        n = vocab.size()[0]
        self.logger.info("Distinct Words: " + str(n))

        filesize = os.stat(cooccurrence_file).st_size
        assert(filesize % 16 == 0)
        nnz = filesize // 16
        self.logger.info("Number of non-zeros: " + str(nnz))

        dt = np.dtype([("ind", "2<i4"), ("val", "<d")])
        data = np.fromfile(cooccurrence_file, dtype=dt)
        ind = torch.IntTensor(data["ind"].transpose()).type(torch.LongTensor) - 1
        val = self.CpuTensor(data["val"])
        cooccurrence = tensor_type.to_sparse(self.CpuTensor)(ind, val, torch.Size([n, n]))
        # TODO: coalescing is very slow, and the cooccurrence matrix is
        # almost always coalesced, but this might not be safe
        # cooccurrence = cooccurrence.coalesce()
        self.logger.info("Loading cooccurrence matrix took " + str(time.time() - begin))

        if initial_vectors is None:
            begin = time.time()
            # TODO: this initialization is really bad for sgd and glove
            if self.embedgpu:
                vectors = tensor_type.to_gpu(self.CpuTensor)(n, self.dim)
            else:
                vectors = self.CpuTensor(n, self.dim)
            vectors.random_(2)
            self.logger.info("Random initialization took " + str(time.time() - begin))
            vectors, _ = util.normalize(vectors)
        else:
            # TODO: verify that the vectors have the right set of words
            # verify that the vectors have a matching dim
            begin = time.time()
            # TODO: select proper precision
            dtype = collections.defaultdict(lambda: self.CpuTensor().numpy().dtype)
            dtype[0] = str
            vectors = pandas.read_csv(initial_vectors, sep=" ", header=None, dtype=dtype).iloc[:, 1:].as_matrix()
            if self.embedgpu:
                vectors = tensor_type.to_gpu(self.CpuTensor)(vectors)
            else:
                vectors = self.CpuTensor(vectors)
            self.logger.info("Loading initial vectors took " + str(time.time() - begin))

        if self.gpu and not self.embedgpu:
            vectors = vectors.pin_memory()

        if initial_bias is not None:
            # TODO: merge this with init bias in glove

            # TODO: verify that the biases have the right set of words
            begin = time.time()
            # TODO: select proper precision
            dtype = collections.defaultdict(lambda: self.CpuTensor().numpy().dtype)
            dtype[0] = str
            self.bias = pandas.read_csv(initial_bias, sep=" ", header=None, dtype=dtype).iloc[:, 1].as_matrix()
            if self.embedgpu: # TODO: own flag?
                self.bias = tensor_type.to_gpu(self.CpuTensor)(self.bias)
            else:
                self.bias = self.CpuTensor(self.bias)
            self.logger.info("Loading initial biases took " + str(time.time() - begin))
        else:
            self.bias = None

        self.load(cooccurrence, vocab, words, vectors)

    def preprocessing(self, mode="ppmi"):
        begin = time.time()

        if self.matgpu:
            self.mat = self.cooccurrence.cuda()
        else:
            self.mat = self.cooccurrence.clone()

        if mode == "none":
            pass
        elif mode == "log1p":
            self.mat._values().log1p_()
        elif mode == "ppmi":
            a = time.time()

            wc = util.sum_rows(self.mat)

            D = torch.sum(wc)  # total dictionary size

            # TODO: pytorch doesn't seem to only allow indexing by 2D tensor
            wc0 = wc[self.mat._indices()[0, :]].squeeze()
            wc1 = wc[self.mat._indices()[1, :]].squeeze()

            ind = self.mat._indices()
            v = self.mat._values()
            nnz = v.shape[0]
            v = torch.log(v) + math.log(D) - torch.log(wc0) - torch.log(wc1)
            v = v.clamp(min=0)

            keep = v.nonzero().squeeze(1)
            if keep.shape[0] != v.shape[0]:
                ind = ind[:, keep]
                v = v[keep]
                self.logger.info("nnz after ppmi processing: " + str(keep.shape[0]))

                self.mat = type(self.mat)(ind, v, torch.Size([self.n, self.n]))
            # self.mat = self.mat.coalesce()

        if self.gpu and not self.matgpu:
            ind = self.mat._indices().t().pin_memory().t()
            v = self.mat._values().pin_memory()
            if self.mat.is_cuda:
                self.mat = tensor_type.to_gpu(tensor_type.to_sparse(self.CpuTensor))(ind, v, torch.Size([self.n, self.n]))
            else:
                self.mat = tensor_type.to_sparse(self.CpuTensor)(ind, v, torch.Size([self.n, self.n]))

        self.logger.info("Preprocessing took " + str(time.time() - begin))

    def solve(self, mode="pi", gpu=True, scale=0.5, normalize=True, iterations=50, eta=1e-3, momentum=0., normfreq=1, batch=100000, innerloop=10):
        if momentum == 0.:
            prev = None
        else:
            if self.embedding.is_cuda:
                prev = tensor_type.to_gpu(self.CpuTensor)(self.n, self.dim)
            else:
                prev = self.CpuTensor(self.n, self.dim)
            prev.zero_()

        if mode == "pi":
            self.embedding, _ = solver.power_iteration(self.mat, self.embedding, x0=prev, iterations=iterations, beta=momentum, norm_freq=normfreq, gpu=gpu)
        elif mode == "alecton":
            self.embedding = solver.alecton(self.mat, self.embedding, iterations=iterations, eta=eta, norm_freq=normfreq, batch=batch)
        elif mode == "vr":
            self.embedding, _ = solver.vr(self.mat, self.embedding, x0=prev, iterations=iterations, beta=momentum, norm_freq=normfreq, batch=batch, innerloop=innerloop)
        elif mode == "sgd":
            self.embedding = solver.sgd(self.mat, self.embedding, iterations=iterations, eta=eta, batch=batch)
        elif mode == "glove":
            # TODO: fix defaults
            # scale = 0
            self.embedding, bias = solver.glove(self.mat, self.embedding, bias=self.bias, iterations=iterations, eta=eta, batch=batch)
        elif mode == "sparsesvd":
            self.embedding = solver.sparseSVD(self.mat, self.dim)
        elif mode == "gensim":
            self.embedding = solver.glove(self.mat)

        self.scale(scale)
        if normalize:
            self.normalize_embeddings()

        if self.embedding.is_cuda:
            begin = time.time()
            self.embedding = self.embedding.cpu()
            self.logger.info("CPU Loading: " + str(time.time() - begin))

    def scale(self, p=1.):
        if p != 0:
            # TODO: Assumes that matrix is normalized.
            begin = time.time()

            # TODO: faster estimation of eigenvalues?
            temp = util.mm(self.mat, self.embedding, self.gpu)
            norm = torch.norm(temp, 2, 0, True)

            norm = norm.pow(p)
            self.embedding = self.embedding.mul(norm.expand_as(self.embedding))
            self.logger.info("Final scaling: " + str(time.time() - begin))

    def normalize_embeddings(self):
        norm = torch.norm(self.embedding, 2, 1, True)
        self.embedding = self.embedding.div(norm.expand_as(self.embedding))

    def save_to_file(self, filename):
        begin = time.time()
        with open(filename, "w") as f:
            for i in range(self.n):
                f.write(self.words[i] + " " + " ".join([str(self.embedding[i, j]) for j in range(self.dim)]) + "\n")
        self.logger.info("Saving embeddings: " + str(time.time() - begin))

if __name__ == "__main__":
    main(sys.argv)
