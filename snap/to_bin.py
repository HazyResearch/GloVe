#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import struct

if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <filename>")
    sys.exit(1)

src = sys.argv[1]
root, ext = os.path.splitext(src)
assert(ext == ".txt")
dest = root + ".bin"

contains = set()
tup = []
with open(src, "r") as src:
        for i in range(4):
            print(src.readline())

        for line in src:
            l = line.split()
            assert(len(l) == 2)
            r = int(l[0])
            c = int(l[1])
            tup.append((r, c))
            contains.add(r)
            contains.add(c)

n = len(contains)
print(n)
m = {}
i = 1
while len(contains) != 0:
    v = contains.pop()
    m[v] = i
    i += 1

with open(dest, "wb") as dest:
    for (r, c) in tup:
        dest.write(struct.pack("iid", m[r], m[c], 1))

with open(root + "-vocab.txt", "w") as vocab:
    for i in range(n):
        vocab.write("vocab_" + str(i) + " 0\n")
