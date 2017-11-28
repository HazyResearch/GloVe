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

method = [
          ["Power Iteration",                "pi",                        1, (0.00, 0.00, 1.00), "-"],
          ["Power Iteration with Momentum",  "pim",                       1, (0.00, 1.00, 1.00), "-"],
        # ["Alecton 1",                      "alecton.ele.606665.0001", 100, "r"],
        # ["Alecton 2",                      "alecton.ele.6066647.001",  10, "r"],
        # ["Alecton 3",                      "alecton.ele.30333233.005",  2, "r"],
        # ["VR 1",                           "vr.ele.606665.00001",       1, "m"],
        # ["VR 2",                           "vr.row.713.00001",          1, "m"],
        # ["VR 3",                           "vr.col.713.00001",          1, "m"],
        # ["VR 1",                           "vr.ele.606665.00001",       1, "m"],
        # ["VR 2",                           "vr.row.713.000001",         1, "m"],
        # ["VR 3",                           "vr.col.713.000001",         1, "m"],
        # ["VR 1",                           "vr.ele.606665.00001",       1, "m"],
        # ["VR ele 1",                       "vr.ele.606665.001.0",       1, "m"],
        # ["VR row 1",                       "vr.row.713.001.0",          1, "m"],
        # ["VR row 1",                       "vr.col.713.001.0",          1, "m"],
        # ["VR ele 2",                       "vr.ele.606665.0001.0",      1, "m"],
        # ["VR row 2",                       "vr.row.713.0001.0",         1, "m"],
        # ["VR row 2",                       "vr.col.713.0001.0",         1, "m"],
        # ["VR ele 3",                       "vr.ele.606665.00001.0",     1, "m"],
        # ["VR row 3",                       "vr.row.713.00001.0",        1, "m"],
        # ["VR row 3",                       "vr.col.713.00001.0",        1, "m"],

          ["Alecton Element (1%, 0.001)",   "alecton.ele.606665.0001",  100, (0.00, 1.00, 0.00), ":"],
          ["Alecton Row (1%, 0.001)",       "alecton.row.713.0001",     100, (0.00, 1.00, 0.00), "--"],
          ["Alecton Column (1%, 0.001)",    "alecton.col.713.0001",     100, (0.00, 1.00, 0.00), "-"],

          ["Alecton Element (1%, 0.0001)",  "alecton.ele.606665.00001", 100, (0.00, 0.50, 0.00), ":"],
          ["Alecton Row (1%, 0.0001)",      "alecton.row.713.00001",    100, (0.00, 0.50, 0.00), "--"],
          ["Alecton Column (1%, 0.0001)",   "alecton.col.713.00001",    100, (0.00, 0.50, 0.00), "-"],

          ["VR Element (1%, 0.00001)",        "vr.ele.606665.00001.0",    1, (1.00, 0.00, 0.00), ":"],
          ["VR Row (1%, 0.00001)",            "vr.row.713.00001.0",       1, (1.00, 0.00, 0.00), "--"],
          ["VR Column (1%, 0.00001)",         "vr.col.713.00001.0",       1, (1.00, 0.00, 0.00), "-"],

          ["VR Element (1%, 0.000001)",        "vr.ele.606665.000001.0",  1, (0.75, 0.00, 0.00), ":"],
          ["VR Row (1%, 0.000001)",            "vr.row.713.000001.0",     1, (0.75, 0.00, 0.00), "--"],
          ["VR Column (1%, 0.000001)",         "vr.col.713.000001.0",      1, (0.75, 0.00, 0.00), "-"],

          ["VR Element (1%, 0.0000001)",        "vr.ele.606665.0000001.0",  1, (0.50, 0.00, 0.00), ":"],
          ["VR Row (1%, 0.0000001)",            "vr.row.713.0000001.0",     1, (0.50, 0.00, 0.00), "--"],
          ["VR Column (1%, 0.0000001)",         "vr.col.713.0000001.0",      1, (0.50, 0.00, 0.00), "-"],
         ]

l1 = {} # First component loss
l2 = {} # Second component loss
lw = {} # Worst component loss
ll = {} # Last component loss

ITERATION = [i + 1 for i in range(10)] + [i for i in range(20, 201, 10)]

for j in range(len(method)):
    m = method[j][0]

    it = [i * method[j][2] for i in ITERATION]

    e = embedding.Embedding(gpu=False)

    l1[m] = []
    l2[m] = []
    lw[m] = []
    ll[m] = []
    for i in it:
        try:
            e.load_vectors("output/" + method[j][1] + "." + str(i) + ".bin")
        except:
            e.load_vectors("output/" + method[j][1] + "." + str(i) + ".txt")
        e.embedding /= e.embedding.norm(2, 0).expand_as(e.embedding)

        l1[m].append(1 - abs(torch.dot(ref.embedding[:, 0], e.embedding[:, 0])))
        l2[m].append(1 - abs(torch.dot(ref.embedding[:, 1], e.embedding[:, 1])))
        lw[m].append(1 - abs(min([torch.dot(ref.embedding[:, i], e.embedding[:, i]) for i in range(dim)])))
        ll[m].append(1 - abs(torch.dot(ref.embedding[:, -1], e.embedding[:, -1])))

plt.figure(1)
for j in range(len(method)):
    m = method[j][0]
    plt.semilogy(ITERATION, l1[m], label=method[j][0], color=method[j][3], linestyle=method[j][4])
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Estimation of First Eigenvector")
plt.savefig("first.png", dpi=300)

plt.figure(2)
for j in range(len(method)):
    m = method[j][0]
    if "VR" in m:
        plt.semilogy(ITERATION[:100], l2[m][:100], label=method[j][0], color=method[j][3], linestyle=method[j][4])
    else:
        plt.semilogy(ITERATION, l2[m], label=method[j][0], color=method[j][3], linestyle=method[j][4])
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Estimation of Second Eigenvector")
plt.savefig("second.png", dpi=300)

plt.figure(3)
for j in range(len(method)):
    m = method[j][0]
    plt.semilogy(ITERATION, lw[m], label=method[j][0], color=method[j][3], linestyle=method[j][4])
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Estimation of Worst Eigenvector")
plt.savefig("worst.png", dpi=300)

plt.figure(4)
for j in range(len(method)):
    m = method[j][0]
    plt.semilogy(ITERATION, ll[m], label=method[j][0], color=method[j][3], linestyle=method[j][4])
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Estimation of Last (" + str(ref.embedding.shape[1]) + ") Eigenvector")
plt.savefig("last.png", dpi=300)
