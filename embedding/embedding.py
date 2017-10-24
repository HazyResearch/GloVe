from __future__ import print_function, absolute_import

import torch
import numpy as np
import time
import os
import struct
import argparse
import sys
import subprocess

import embedding.solver as solver
import embedding.util as util
import embedding.evaluate as evaluate

def main(argv=None):

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Tools for embeddings.")
    subparser = parser.add_subparsers(dest="task")

    # Preprocessing parser
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
    compute_parser.add_argument("-n", "--normalize", type=str2bool, default=True,
                                help="Toggle to normalize embeddings")

    compute_parser.add_argument("-g", "--gpu", type=str2bool, default=True,
                                help="Toggle to use GPU")

    # Evaluate parser
    evaluate_parser = subparser.add_parser("evaluate", help="Evaluate performance of an embedding on standard tasks.")

    evaluate_parser.add_argument('--vocab', type=str, default='vocab.txt',
                                 help="filename of vocabulary file")
    evaluate_parser.add_argument('--vectors', type=str, default='vectors.txt',
                                 help="filename of embedding vectors file")

    args = parser.parse_args(argv)

    if hasattr(args, "gpu") and args.gpu and not torch.cuda.is_available():
        print("WARNING: GPU use requested, but GPU not available.")
        print("         Toggling off GPU use.")
        sys.stdout.flush()
        args.gpu = False

    if args.task == "cooccurrence":
        # subprocess.call(["./cooccurrence.sh", args.text], cwd=os.path.join(os.path.dirname(__file__), "..",))
        subprocess.call([os.path.join(os.path.dirname(__file__), "..", "cooccurrence.sh"), args.text])
    elif args.task == "compute":
        embedding = Embedding(args.dim)
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
        evaluate.evaluate_vectors_analogy(W_norm, vocab, ivocab)
        # evaluate.evaluate_human_sim()
        evaluate.evaluate_vectors_sim(W, vocab, ivocab)


class Embedding(object):
    def __init__(self, dim=50):
        self.dim = dim

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
            vocab = torch.DoubleTensor([l[1] for l in lines])
        n = vocab.size()[0]
        print("n:", n)
        sys.stdout.flush()

        filesize = os.stat(cooccurrence_file).st_size
        assert(filesize % 16 == 0)
        nnz = filesize // 16
        print("nnz:", nnz)
        sys.stdout.flush()
        v = np.empty(nnz, np.float64)
        ind = np.empty((2, nnz), np.int64) # TODO: binary format is int32, but torch uses Long
        with open(cooccurrence_file, "rb") as f:
            content = f.read()
            i = 0
            block = 10000
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

        if initial_vectors is None:
            vectors = torch.randn([n, self.dim]).type(torch.DoubleTensor)
        else:
            # TODO: verify that the vectors have the right set of words
            # verify that the vectors have a matching dim
            with open(initial_vectors, "r") as f:
                vectors = torch.DoubleTensor([[float(v) for v in line.split()[1:]] for line in f])
        vectors, _ = util.normalize(vectors)

        self.load(cooccurrence, vocab, words, vectors)

        end = time.time()
        print("Loading data took", end - begin)
        sys.stdout.flush()

    def preprocessing(self, mode="ppmi"):
        begin = time.time()
        if mode == "none":
            self.mat = self.cooccurrence
        elif mode == "log1p":
            self.mat = self.cooccurrence.clone()
            self.mat._values().log1p_()
        elif mode == "ppmi":
            self.mat = self.cooccurrence.clone()
            wc = torch.mm(self.mat, torch.ones([self.n, 1]).type(torch.DoubleTensor)) # individual word counts
            D = torch.sum(wc) # total dictionary size
            # TODO: pytorch doesn't seem to only allow indexing by vector
            wc0 = wc[self.mat._indices()[0, :]].squeeze()
            wc1 = wc[self.mat._indices()[1, :]].squeeze()

            ind = self.mat._indices()
            v = self.mat._values()
            nnz = v.shape[0]
            v = torch.log(v) + torch.log(torch.DoubleTensor(nnz).fill_(D)) - torch.log(wc0) - torch.log(wc1)
            v = v.clamp(min=0)
            self.mat = torch.sparse.DoubleTensor(ind, v, torch.Size([self.n, self.n])).coalesce()
        end = time.time()
        print("Preprocessing took", end - begin)
        sys.stdout.flush()

    def solve(self, mode="pi", gpu=True, scale=0.5, normalize=True, iterations=50, eta=1e-3, momentum=0., normfreq=1, batch=100000, innerloop=10):
        if momentum == 0.:
            prev = None
        else:
            prev = torch.zeros([self.n, self.dim]).type(torch.DoubleTensor)

        if gpu:
            begin = time.time()
            self.mat = self.mat.cuda()
            self.embedding = self.embedding.cuda()
            if prev is not None:
                prev = prev.cuda()
            end = time.time()
            print("GPU Loading:", end - begin)
            sys.stdout.flush()

        if mode == "pi":
            self.embedding, _ = solver.power_iteration(self.mat, self.embedding, x0=prev, iterations=iterations, beta=momentum, norm_freq=normfreq)
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
