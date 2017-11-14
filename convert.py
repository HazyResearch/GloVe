#!/usr/bin/env python

import embedding
import os
import pathlib

embed = embedding.Embedding(gpu=False)
embed.load_cooccurrence()

for filename in os.listdir("output"):
    filename = "output/" + filename
    root, ext = os.path.splitext(filename)
    if ext == ".txt":
        binname = root + ".bin"
        print(filename)
        if (not pathlib.Path(binname).is_file() or
            os.stat(binname).st_mtime < os.stat(filename).st_mtime):
            embed.load_vectors(filename)
            embed.save_vectors(binname)

