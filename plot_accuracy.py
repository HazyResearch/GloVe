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

import seaborn as sns

import embedding

sns.set(style="whitegrid", color_codes=True)

embed = embedding.Embedding(gpu=False)
embed.load_cooccurrence()

def load_eval(root):
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

method = {
          "Power Iteration": "pi",
          "Power Iteration with Momentum": ["pim", 1],
          "Alecton 1": "alecton.ele.606665.0001",
          "Alecton 2": "alecton.ele.6066647.001",
          "Alecton 3": "alecton.ele.30333233.005",
          "VR 1": "vr.ele.606665.00001",
          "VR 2": "vr.row.713.00001",
          "VR 3": "vr.col.713.00001",
          "VR 1": "vr.ele.606665.00001",
          "VR 2": "vr.row.713.000001",
          "VR 3": "vr.col.713.000001",
          "VR 1": "vr.ele.606665.00001",
          "VR ele 1": "vr.ele.606665.001.0",
          "VR row 1": "vr.row.713.001.0",
          "VR row 1": "vr.col.713.001.0",
          "VR ele 2": "vr.ele.606665.0001.0",
          "VR row 2": "vr.row.713.0001.0",
          "VR row 2": "vr.col.713.0001.0",
          "VR ele 3": "vr.ele.606665.00001.0",
          "VR row 3": "vr.row.713.00001.0",
          "VR row 3": "vr.col.713.00001.0",
          "VR ele 4": "vr.ele.606665.000001.0",
          "VR row 4": "vr.row.713.000001.0",
          "VR row 4": "vr.col.713.000001.0",
         }

score = collections.defaultdict(lambda : collections.defaultdict(list))

for m in method:
    it = [i + 1 for i in range(100)]
    for i in it:
        e = load_eval("output/" + method[m] + "." + str(i))
        for task in e:
            score[task][m].append(e[task])

for (i, task) in enumerate(score):
    plt.figure(i)
    for m in method:
        plt.plot([0] + it, [0] + score[task][m], label=m)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title(task)
    plt.savefig(task + ".pdf", dpi=300)
