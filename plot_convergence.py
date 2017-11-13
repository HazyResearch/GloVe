#!/usr/bin/env python

import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

import embedding

sns.set(style="whitegrid", color_codes=True)

ref = embedding.Embedding(gpu=False)
ref.load_vectors("output/pi.5000.txt")
ref.embedding /= ref.embedding.norm(2, 0).expand_as(ref.embedding)
dim = ref.embedding.shape[1]

method = {
          "Power Iteration": ["pi", 1],
          "Power Iteration with Momentum": ["pim", 1],
          # "Alecton 1": ["alecton.ele.606665.0001", 100],
          # "Alecton 2": ["alecton.ele.6066647.001", 10],
          # "Alecton 3": ["alecton.ele.30333233.005", 2],
          # "VR 1": ["vr.ele.606665.00001", 1],
          # "VR 2": ["vr.row.713.00001", 1],
          # "VR 3": ["vr.col.713.00001", 1],
          "VR 1": ["vr.ele.606665.00001", 1],
          # "VR 2": ["vr.row.713.000001", 1],
          # "VR 3": ["vr.col.713.000001", 1],
         }

l1 = {} # First component loss
l2 = {} # Second component loss
lw = {} # Worst component loss
ll = {} # Last component loss

ITERATION = [i + 1 for i in range(100)]

for m in method:

    it = [i * method[m][1] for i in ITERATION]

    e = embedding.Embedding(gpu=False)

    l1[m] = []
    l2[m] = []
    lw[m] = []
    ll[m] = []
    for i in it:
        e.load_vectors("output/" + method[m][0] + "." + str(i) + ".txt")
        e.embedding /= e.embedding.norm(2, 0).expand_as(e.embedding)

        l1[m].append(1 - abs(torch.dot(ref.embedding[:, 0], e.embedding[:, 0])))
        l2[m].append(1 - abs(torch.dot(ref.embedding[:, 1], e.embedding[:, 1])))
        lw[m].append(1 - abs(min([torch.dot(ref.embedding[:, i], e.embedding[:, i]) for i in range(dim)])))
        ll[m].append(1 - abs(torch.dot(ref.embedding[:, -1], e.embedding[:, -1])))

plt.figure(1)
for m in method:
    plt.semilogy(ITERATION, l1[m], label=m)
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Estimation of First Eigenvector")
plt.savefig("first.pdf", dpi=300)

plt.figure(2)
for m in method:
    plt.semilogy(ITERATION, l2[m], label=m)
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Estimation of Second Eigenvector")
plt.savefig("second.pdf", dpi=300)

plt.figure(3)
for m in method:
    plt.semilogy(ITERATION, lw[m], label=m)
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Estimation of Worst Eigenvector")
plt.savefig("worst.pdf", dpi=300)

plt.figure(4)
for m in method:
    plt.semilogy(ITERATION, ll[m], label=m)
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Estimation of Last Eigenvector")
plt.savefig("last.pdf", dpi=300)
