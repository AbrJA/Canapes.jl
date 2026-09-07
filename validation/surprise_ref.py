#!/usr/bin/env python3
# validation/surprise_ref.py — Surprise reference predictions for the explicit
# subsystem (validate_surprise.jl).
#
# Fits: SlopeOne, BaselineOnly (ALS), KNNWithMeans (pearson), SVD (biased),
# on the fixture matrix written by validate_surprise.jl and dumps the held-out
# predictions per model to CSV in the same directory.
import os
import sys

import numpy as np
from surprise import (
    BaselineOnly,
    Dataset,
    KNNWithMeans,
    Reader,
    SlopeOne,
    SVD,
)

fdir = sys.argv[1] if len(sys.argv) > 1 else "."
train_path = os.path.join(fdir, "train.csv")
test_path = os.path.join(fdir, "test.csv")

reader = Reader(line_format="user item rating", sep="\t", rating_scale=(1, 5), skip_lines=1)
train_data = Dataset.load_from_file(train_path, reader=reader).build_full_trainset()
test_entries = []
with open(test_path) as fh:
    next(fh)
    for line in fh:
        u, i, r = line.split()
        test_entries.append((int(u), int(i), float(r)))

rng = np.random.RandomState(42)


def dump(trainset, algo, name, entries):
    algo.fit(trainset)
    with open(os.path.join(fdir, f"surprise_{name}.csv"), "w") as fh:
        fh.write("user,item,pred\n")
        for u, i, _r in entries:
            p = algo.predict(str(u), str(i), verbose=False).est
            fh.write(f"{u},{i},{p:.10f}\n")


dump(train_data, SlopeOne(), "slopeone", test_entries)

dump(
    train_data,
    BaselineOnly(
        bsl_options=dict(method="als", n_epochs=15, reg_u=0.02, reg_i=0.02),
        verbose=False,
    ),
    "baseline",
    test_entries,
)

dump(
    train_data,
    KNNWithMeans(k=20, min_k=1, sim_options=dict(name="pearson"), verbose=False),
    "pearson",
    test_entries,
)

dump(
    train_data,
    SVD(n_factors=12, n_epochs=40, lr_all=0.01, reg_all=0.05, biased=True, random_state=42),
    "svd",
    test_entries,
)

print("surprise references written:", ["slopeone", "baseline", "pearson", "svd"])