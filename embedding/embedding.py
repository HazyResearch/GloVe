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

import embedding.solver as solver
import embedding.util as util
import embedding.evaluate as evaluate

def main(argv=None):

    parser = argparse.ArgumentParser(description="Tools for embeddings.")
    subparser = parser.add_subparsers(dest="task")

    # Cooccurrence parser
    cooccurrence_parser = subparser.add_parser("cooccurrence", help="Preprocessing (compute vocab and cooccurrence from text.")

    cooccurrence_parser.add_argument("text", type=str, nargs="?", default="text",
                                   help="filename of text file")

    # Compute parser
    compute_parser = subparser.add_parser("compute", help="Compute embedding from scratch via cooccurrence matrix.")

    compute_parser.add_argument("-d", "--dim", type=int, default=50,
                                help="dimension of embedding")

    compute_parser.add_argument("-v", "--vocab", type=str, default="vocab.txt",
                                help="filename of vocabulary file")
    compute_parser.add_argument("-c", "--cooccurrence", type=str, default="cooccurrence.shuf.bin",
                                help="filename of cooccurrence binary")
    compute_parser.add_argument("--initial", type=str, default=None,
                                help="filename of initial embedding vectors")
    compute_parser.add_argument("-o", "--vectors", type=str, default="vectors.txt",
                                help="filename for embedding vectors output")

    compute_parser.add_argument("-p", "--preprocessing", type=str, default="ppmi",
                                choices=["none", "log1p", "ppmi"],
                                help="Preprocessing of cooccurrence matrix before eigenvector computation")

    compute_parser.add_argument("-s", "--solver", type=str, default="pi",
                                choices=["pi", "alecton", "vr", "sgd"],
                                help="Solver used to find top eigenvectors")
    compute_parser.add_argument("-i", "--iterations", type=int, default=50,
                                help="Iterations used by solver")
    compute_parser.add_argument("-e", "--eta", type=float, default=1e-3,
                                help="Learning rate used by solver")
    compute_parser.add_argument("-m", "--momentum", type=float, default=0.,
                                help="Momentum used by solver")
    compute_parser.add_argument("-f", "--normfreq", type=int, default=1,
                                help="Normalization frequency used by solver")
    compute_parser.add_argument("-b", "--batch", type=int, default=100000,
                                help="Batch size used by solver")
    compute_parser.add_argument("-j", "--innerloop", type=int, default=10,
                                help="Inner loop iterations used by solver")

    compute_parser.add_argument("--scale", type=float, default=0.5,
                                help="Scale on eigenvector is $\lambda_i ^ s$")
    compute_parser.add_argument("-n", "--normalize", type=util.str2bool, default=True,
                                help="Toggle to normalize embeddings")

    compute_parser.add_argument("-g", "--gpu", type=util.str2bool, default=True,
                                help="Toggle to use GPU")

    compute_parser.add_argument("--precision", type=str, default="float",
                                choices=["half", "float", "double"],
                                help="Precision of values")

    # Evaluate parser
    evaluate_parser = subparser.add_parser("evaluate", help="Evaluate performance of an embedding on standard tasks.")

    evaluate_parser.add_argument('--vocab', type=str, default='vocab.txt',
                                 help="filename of vocabulary file")
    evaluate_parser.add_argument('--vectors', type=str, default='vectors.txt',
                                 help="filename of embedding vectors file")

    args = parser.parse_args(argv)


    if args.task == "cooccurrence":
        # subprocess.call(["./cooccurrence.sh", args.text], cwd=os.path.join(os.path.dirname(__file__), "..",))
        subprocess.call([os.path.join(os.path.dirname(__file__), "..", "cooccurrence.sh"), args.text])
    elif args.task == "compute":
        if args.gpu and not torch.cuda.is_available():
            print("WARNING: GPU use requested, but GPU not available.")
            print("         Toggling off GPU use.")
            sys.stdout.flush()
            args.gpu = False

        CpuTensor = torch.FloatTensor
        GpuTensor = torch.cuda.FloatTensor
        CpuSparseTensor = torch.sparse.FloatTensor
        GpuSparseTensor = torch.cuda.sparse.FloatTensor
        if args.precision == "half":
            CpuTensor = torch.HalfTensor
            GpuTensor = torch.cuda.HalfTensor
            CpuSparseTensor = torch.sparse.HalfTensor
            GpuSparseTensor = torch.cuda.sparse.HalfTensor
        elif args.precision == "float":
            CpuTensor = torch.FloatTensor
            GpuTensor = torch.cuda.FloatTensor
            CpuSparseTensor = torch.sparse.FloatTensor
            GpuSparseTensor = torch.cuda.sparse.FloatTensor
        elif args.precision == "double":
            CpuTensor = torch.DoubleTensor
            GpuTensor = torch.cuda.DoubleTensor
            CpuSparseTensor = torch.sparse.DoubleTensor
            GpuSparseTensor = torch.cuda.sparse.DoubleTensor
        else:
            print("WARNING: Precision \"" + args.precision + "\" is not recognized.")
            print("         Defaulting to \"float\".")
            sys.stdout.flush()

        embedding = Embedding(args.dim, args.gpu, CpuTensor, GpuTensor, CpuSparseTensor, GpuSparseTensor)
        embedding.load_from_file(args.vocab, args.cooccurrence, args.initial)
        # embedding.load(*util.synthetic(2, 4))
        embedding.preprocessing(args.preprocessing)
        embedding.solve(mode=args.solver, gpu=args.gpu, scale=args.scale, normalize=args.normalize, iterations=args.iterations, eta=args.eta, momentum=args.momentum, normfreq=args.normfreq, batch=args.batch, innerloop=args.innerloop)
        embedding.save_to_file(args.vectors)
    elif args.task == "evaluate":
        with open(args.vocab, 'r') as f:
            words = [x.rstrip().split(' ')[0] for x in f.readlines()]
        with open(args.vectors, 'r') as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                vectors[vals[0]] = [float(x) for x in vals[1:]]

        vocab_size = len(words)
        vocab = {w: idx for idx, w in enumerate(words)}
        ivocab = {idx: w for idx, w in enumerate(words)}

        vector_dim = len(vectors[ivocab[0]])
        W = np.zeros((vocab_size, vector_dim))
        for word, v in vectors.items():
            if word == '<unk>':
                continue
            W[vocab[word], :] = v

        # normalize each word vector to unit variance
        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T
        # evaluate.evaluate_human_sim()
        evaluate.evaluate_vectors_sim(W, vocab, ivocab)
        evaluate.evaluate_vectors_analogy(W_norm, vocab, ivocab)


class Embedding(object):
    def __init__(self, dim=50, gpu=True, CpuTensor=torch.FloatTensor, GpuTensor=torch.cuda.FloatTensor,
                 CpuSparseTensor=torch.cuda.FloatTensor, GpuSparseTensor=torch.cuda.sparse.FloatTensor):
        self.dim = dim
        self.gpu = gpu
        self.CpuTensor = CpuTensor
        self.GpuTensor = GpuTensor
        self.CpuSparseTensor = CpuSparseTensor
        self.GpuSparseTensor = GpuSparseTensor

    def load(self, cooccurrence, vocab, words, embedding=None):
        self.n = cooccurrence.size()[0]

        # TODO: error if n > dim

        self.cooccurrence = cooccurrence
        self.vocab = vocab
        self.words = words
        self.embedding = embedding

    def load_from_file(self, vocab_file="vocab.txt", cooccurrence_file="cooccurrence.shuf.bin", initial_vectors=None):
        # TODO: option of FloatTensor

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
        print("n:", n)
        sys.stdout.flush()

        filesize = os.stat(cooccurrence_file).st_size
        assert(filesize % 16 == 0)
        nnz = filesize // 16
        print("nnz:", nnz)
        sys.stdout.flush()

        dt = np.dtype([("ind", [("row", "<i4"), ("col", "<i4")]), ("val", "<d")])
        dt = np.dtype([("row", "<i4"), ("col", "<i4"), ("val", "<d")])
        data = np.fromfile(cooccurrence_file, dtype=dt)
        ind = torch.IntTensor(np.array([data["row"], data["col"]])).type(torch.LongTensor) - 1
        val = self.CpuTensor(data["val"])
        cooccurrence = self.CpuSparseTensor(ind, val, torch.Size([n, n]))
        # TODO: avoid putting cooccurrence on gpu?
        # delay until preprocessing
        if self.gpu:
            cooccurrence = cooccurrence.cuda()
        # TODO: coalescing is very slow, and the cooccurrence matrix is
        # almost always coalesced, but this might not be safe
        # cooccurrence = cooccurrence.coalesce()
        end = time.time()
        print("Loading data took", end - begin)
        sys.stdout.flush()

        if initial_vectors is None:
            begin = time.time()
            if self.gpu:
                vectors = self.GpuTensor(n, self.dim)
            else:
                vectors = self.CpuTensor(n, self.dim)
            vectors.random_(2)
            end = time.time()
            print("Random initialization took ", end - begin)
            vectors, _ = util.normalize(vectors)
        else:
            # TODO: verify that the vectors have the right set of words
            # verify that the vectors have a matching dim
            with open(initial_vectors, "r") as f:
                if self.gpu:
                    vectors = self.GpuTensor([[float(v) for v in line.split()[1:]] for line in f])
                else:
                    vectors = self.CpuTensor([[float(v) for v in line.split()[1:]] for line in f])

        self.load(cooccurrence, vocab, words, vectors)

    def preprocessing(self, mode="ppmi"):
        begin = time.time()
        if mode == "none":
            self.mat = self.cooccurrence
        elif mode == "log1p":
            self.mat = self.cooccurrence.clone()
            self.mat._values().log1p_()
        elif mode == "ppmi":
            a = time.time()
            self.mat = self.cooccurrence.clone()
            # TODO: better way of getting row/col sum
            if self.mat.is_cuda:
                wc = torch.mm(self.mat, torch.ones([self.n, 1]).type(self.GpuTensor)) # individual word counts
            else:
                wc = torch.mm(self.mat, torch.ones([self.n, 1]).type(self.CpuTensor)) # individual word counts
            D = torch.sum(wc) # total dictionary size
            # TODO: pytorch doesn't seem to only allow indexing by vector
            wc0 = wc[self.mat._indices()[0, :]].squeeze()
            wc1 = wc[self.mat._indices()[1, :]].squeeze()

            ind = self.mat._indices()
            v = self.mat._values()
            nnz = v.shape[0]
            v = torch.log(v) + math.log(D) - torch.log(wc0) - torch.log(wc1)
            v = v.clamp(min=0)
            if ind.is_cuda:
                keep = (v != 0).type(torch.cuda.LongTensor)
            else:
                keep = (v != 0).type(torch.LongTensor)
            ind = ind[:, keep]
            v = v[keep]
            print("nnz after ppmi processing:", torch.sum(keep))
            if self.mat.is_cuda:
                self.mat = self.GpuSparseTensor(ind, v, torch.Size([self.n, self.n]))
            else:
                self.mat = self.CpuSparseTensor(ind, v, torch.Size([self.n, self.n]))
            # self.mat = self.mat.coalesce()
        end = time.time()
        print("Preprocessing took", end - begin)
        sys.stdout.flush()

    def solve(self, mode="pi", gpu=True, scale=0.5, normalize=True, iterations=50, eta=1e-3, momentum=0., normfreq=1, batch=100000, innerloop=10):
        if momentum == 0.:
            prev = None
        else:
            if self.embedding.is_cuda:
                prev = self.GpuTensor(self.n, self.dim)
                prev.zeros_()
            else:
                prev = self.CpuTensor(self.n, self.dim)
                prev.zeros_()

        if mode == "pi":
            self.embedding, _ = solver.power_iteration(self.mat, self.embedding, x0=prev, iterations=iterations, beta=momentum, norm_freq=normfreq, gpu = gpu)
        elif mode == "alecton":
            self.embedding = solver.alecton(self.mat, self.embedding, iterations=iterations, eta=eta, norm_freq=normfreq, batch=batch)
        elif mode == "vr":
            self.embedding, _ = solver.vr(self.mat, self.embedding, x0=prev, iterations=iterations, beta=momentum, norm_freq=normfreq, batch=batch, innerloop=innerloop)
        elif mode == "sgd":
            self.embedding = solver.sgd(self.mat, self.embedding, iterations=iterations, eta=eta, norm_freq=normfreq, batch=batch)

        self.scale(scale)
        if normalize:
            self.normalize_embeddings()

        if gpu:
            begin = time.time()
            self.embedding = self.embedding.cpu()
            end = time.time()
            print("CPU Loading:", end - begin)
            sys.stdout.flush()

    def scale(self, p=1.):
        # TODO: Assumes that matrix is normalized.
        begin = time.time()

        # TODO: faster estimation of eigenvalues?
        temp = torch.mm(self.mat, self.embedding)
        norm = torch.norm(temp, 2, 0, True)

        norm = norm.pow(p)
        self.embedding = self.embedding.mul(norm.expand_as(self.embedding))
        end = time.time()
        print("Final scaling:", end - begin)
        sys.stdout.flush()

    def normalize_embeddings(self):
        norm = torch.norm(self.embedding, 2, 1, True)
        self.embedding = self.embedding.div(norm.expand_as(self.embedding))

    def save_to_file(self, filename):
        begin = time.time()
        with open(filename, "w") as f:
            for i in range(self.n):
                f.write(self.words[i] + " " + " ".join([str(self.embedding[i, j]) for j in range(self.dim)]) + "\n")
        end = time.time()
        print("Saving embeddings:", end - begin)
        sys.stdout.flush()

if __name__ == "__main__":
    main(sys.argv)
