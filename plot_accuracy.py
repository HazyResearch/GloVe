#!/usr/bin/env python

import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import pathlib
import os
import collections
from multiprocessing import Pool

import seaborn as sns

import embedding

sns.set(style="whitegrid", color_codes=True)

embed = embedding.Embedding(gpu=False)
embed.load_cooccurrence()

def load_eval(root):
    print("load_eval " + root)
    pkl_filename = root + ".eval.pkl"
    vec_filename = root + ".txt"
    if (pathlib.Path(pkl_filename).is_file() and
        os.stat(pkl_filename).st_mtime > os.stat(vec_filename).st_mtime):
        # TODO: check age of evaluate.py
        with open(pkl_filename, "rb") as f:
            return pickle.load(f)

    embed.load_vectors(vec_filename)
    score = embed.evaluate()
    with open(pkl_filename, "wb") as f:
        pickle.dump(score, f)
    return score

method = [
          ["Power Iteration",               "pi",                        1, (0.00, 0.00, 1.00)],
          ["Power Iteration with Momentum", "pim",                       1, (0.00, 1.00, 1.00)],
          # "Alecton Element (1%, 0.001)",   ["alecton.ele.606665.0001", 100, (0.00, 0.50, 0.00)],
          # "Alecton Row (1%, 0.001)",       ["alecton.row.713.0001",    100, (0.00, 0.50, 0.00)],
          ["Alecton Column (1%, 0.0001)",    "alecton.col.713.0001",    100, (0.00, 1.00, 0.00)],
          ["Alecton Column (1%, 0.00001)",   "alecton.col.713.00001",   100, (0.00, 0.50, 0.00)],
          # "VR Element (1%, 0.0001)",        ["vr.ele.606665.0001.0",       1, (1.00, 0.00, 0.00)],
          # "VR Row (1%, 0.0001)",            ["vr.row.713.0001.0",          1, (1.00, 0.20, 0.00)],
          ["VR Column (1%, 0.00001)",        "vr.col.713.00001.0",          1, (1.00, 0.00, 0.20)],
          # "VR Element (1%, 0.001)",        ["vr.ele.606665.00001.0",       1, (0.50, 0.00, 0.00)],
          # "VR Row (1%, 0.001)",            ["vr.row.713.00001.0",          1, (0.50, 0.20, 0.00)],
          ["VR Column (1%, 0.000001)",       "vr.col.713.000001.0",          1, (0.50, 0.00, 0.20)],
          # "VR ele 2", ["vr.ele.606665.0001.0", 1],
          # "VR row 2", ["vr.row.713.0001.0", 1],
          # "VR col 2", ["vr.col.713.0001.0", 1],
          # "VR ele 3", ["vr.ele.606665.00001.0", 1],
          # "VR row 3", ["vr.row.713.00001.0", 1],
          # "VR col 3", ["vr.col.713.00001.0", 1],
          # "VR ele 4", ["vr.ele.606665.000001.0", 1],
          # "VR row 4", ["vr.row.713.000001.0", 1],
          # "VR col 4", ["vr.col.713.000001.0", 1],
         ]

score = collections.defaultdict(lambda : collections.defaultdict(list))

ITERATION = [i + 1 for i in range(50)]

pool = Pool()
pool.map(load_eval, ["output/" + method[j][1] + "." + str(i * method[j][2]) for i in ITERATION for j in range(len(method))])

for j in range(len(method)):
    m = method[j][0]
    for i in ITERATION:
        e = load_eval("output/" + method[j][1] + "." + str(i * method[j][2]))
        for task in e:
            score[task][m].append(e[task])

for (i, task) in enumerate(score):
    plt.figure(i)
    h = []
    l = []
    for j in range(len(method)):
        m = method[j][0]
        print(ITERATION)
        print(score[task][m])
        h.append(plt.plot([0] + ITERATION, [0] + score[task][m], label=m, color=method[j][3]))
        l.append(m)

    ymax = max([v for s in score[task].values() for v in s])
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.ylim(0, 1.1 * ymax)
    plt.title(task)
    plt.savefig(task + ".pdf", dpi=300)
