#!/usr/bin/env python3
# validation/ml1m_ref.py — implicit reference models on the MovieLens-1M split
# produced by validate_ml1m.jl. Trains ALS, BPR and cosine ItemItem, writes
# per-user top-10 recommendations (1-based) to py_<model>.txt and fit/predict
# timings to py_timings.json.
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender

d = sys.argv[1]
n_users = int(sys.argv[2])
n_items = int(sys.argv[3])


def load(name):
    rows, cols = [], []
    with open(f"{d}/{name}.txt") as fh:
        for line in fh:
            u, i = line.split()
            rows.append(int(u) - 1)
            cols.append(int(i) - 1)
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


train = load("train")
test = load("test")

models = [
    ("als", AlternatingLeastSquares(factors=32, regularization=0.1, alpha=40.0, iterations=10)),
    ("bpr", BayesianPersonalizedRanking(factors=64, iterations=100, learning_rate=0.1, regularization=0.01)),
    ("itemknn", CosineRecommender(K=400)),
]

timings = {}
for name, model in models:
    t0 = time.time()
    model.fit(train)
    fit_s = time.time() - t0
    t0 = time.time()
    rec, _ = model.recommend(np.arange(n_users), train, N=10, filter_already_liked_items=True)
    pred_s = time.time() - t0
    timings[name] = {"fit_s": round(fit_s, 2), "pred_s": round(pred_s, 2)}
    with open(f"{d}/py_{name}.txt", "w") as fh:
        for u in range(n_users):
            fh.write(" ".join(str(x + 1) for x in rec[u]) + "\n")

with open(f"{d}/py_timings.txt", "w") as fh:
    for name, t in timings.items():
        fh.write(f"{name} {t['fit_s']} {t['pred_s']}\n")
print("python ref done:", list(timings))