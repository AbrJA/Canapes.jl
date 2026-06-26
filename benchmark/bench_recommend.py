#!/usr/bin/env python3
"""
benchmark/bench_recommend.py — Python (implicit + sklearn) inference benchmark.

Counterpart to benchmark/bench_recommend.jl for Julia vs Python comparison.
Benchmarks recommend / score / predict throughput and peak memory.

Run:
    python benchmark/bench_recommend.py          # uses system python
    PYTHON=path/to/python benchmark/bench_recommend.py

Requires: implicit, scikit-learn, scipy, numpy, psutil
"""
import os
import sys
import csv
import gc
import time
import tracemalloc
from statistics import median

import numpy as np
import psutil
from scipy.sparse import csr_matrix, random as sp_random
from scipy.sparse.linalg import svds

import implicit

# ─────────────────────────────────────────────
# Memory monitoring
# ─────────────────────────────────────────────

def rss_bytes():
    """Current RSS in bytes."""
    return psutil.Process().memory_info().rss

def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024**2:
        return f"{b / 1024:.1f} KiB"
    if b < 1024**3:
        return f"{b / 1024**2:.1f} MiB"
    return f"{b / 1024**3:.1f} GiB"

def bench(fn, warmup=True, repeats=3):
    """Run fn() multiple times, return (median_time, median_rss_delta)."""
    if warmup:
        fn()

    times = []
    mem_deltas = []
    for _ in range(repeats):
        gc.collect()
        rss_before = rss_bytes()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        rss_after = rss_bytes()
        times.append(elapsed)
        mem_deltas.append(max(0, rss_after - rss_before))

    return median(times), median(mem_deltas)

# ─────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────

def generate_matrix(n_users, n_items, density, seed=42):
    rng = np.random.default_rng(seed)
    nnz_target = int(n_users * n_items * density)
    rows = rng.integers(0, n_users, size=nnz_target)
    cols = rng.integers(0, n_items, size=nnz_target)
    vals = np.ones(nnz_target, dtype=np.float32)
    X = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    X.sum_duplicates()
    return X

def generate_regression_data(n_samples, n_features, density, seed=42):
    rng = np.random.default_rng(seed)
    nnz_target = int(n_samples * n_features * density)
    rows = rng.integers(0, n_samples, size=nnz_target)
    cols = rng.integers(0, n_features, size=nnz_target)
    vals = rng.standard_normal(nnz_target).astype(np.float32)
    X = csr_matrix((vals, (rows, cols)), shape=(n_samples, n_features))
    X.sum_duplicates()
    y = rng.integers(0, 2, size=n_samples).astype(np.float32)
    return X, y

# ─────────────────────────────────────────────
# Scales (must match Julia benchmark)
# ─────────────────────────────────────────────

SCALES = [
    {"name": "small",  "n_users": 1_000,   "n_items": 500,    "density": 0.05},
    {"name": "medium", "n_users": 10_000,  "n_items": 5_000,  "density": 0.01},
    {"name": "large",  "n_users": 100_000, "n_items": 20_000, "density": 0.002},
]

# ─────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────

def _make_bpr(**kw):
    cls = getattr(implicit.bpr, "BPR",
           getattr(implicit.bpr, "BayesianPersonalizedRanking", None))
    if cls is None:
        raise RuntimeError("No BPR class found in implicit")
    return cls(**kw)

def _make_lmf(**kw):
    return implicit.lmf.LogisticMatrixFactorization(**kw)

def mf_models():
    return [
        ("ALS", lambda: implicit.als.AlternatingLeastSquares(
            factors=64, regularization=0.1, alpha=40.0,
            iterations=5, random_state=42, use_gpu=False)),
        ("ALS-CG", lambda: implicit.als.AlternatingLeastSquares(
            factors=64, regularization=0.1, alpha=40.0,
            iterations=5, random_state=42, use_gpu=False)),
        ("BPR", lambda: _make_bpr(
            factors=64, regularization=0.01, learning_rate=0.05,
            iterations=5, random_state=42, use_gpu=False)),
        ("LMF", lambda: _make_lmf(
            factors=64, learning_rate=1.0, regularization=0.6,
            iterations=5, random_state=42)),
    ]

def similarity_models():
    """EASE / ItemKNN via sklearn or numpy — not available in implicit."""
    return [
        ("EASE-np", None),    # implemented inline below
        ("ItemKNN-sk", None), # implemented inline below
    ]

# ─────────────────────────────────────────────
# Pure-Python EASE (closed-form, for inference benchmark)
# ─────────────────────────────────────────────

def ease_fit(X_csr, lam=500.0):
    """Fit EASE: B = I - P·diag(1/diag(P)) where P = (G + λI)^{-1}."""
    X = X_csr.toarray().astype(np.float64)
    G = X.T @ X
    G += lam * np.eye(G.shape[0])
    P = np.linalg.inv(G)
    B = np.eye(P.shape[0]) - P @ np.diag(1.0 / np.diag(P))
    np.fill_diagonal(B, 0.0)
    return B.astype(np.float32)

def ease_recommend(X_csr, B, k=10):
    """Score all users and extract top-k (excluding seen)."""
    scores = X_csr.toarray() @ B
    # mask seen items
    seen = X_csr.toarray() > 0
    scores[seen] = -np.inf
    topk = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    return topk

def ease_score(X_csr, B):
    return X_csr.toarray() @ B

# ─────────────────────────────────────────────
# Pure-Python ItemKNN via sklearn
# ─────────────────────────────────────────────

def knn_fit(X_csr, k=50):
    """Fit ItemKNN: cosine similarity, keep top-k per item."""
    from sklearn.metrics.pairwise import cosine_similarity
    X = X_csr.T.toarray().astype(np.float64)   # items × users
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0.0)
    # keep top-k per column (item)
    W = np.zeros_like(S)
    for j in range(S.shape[1]):
        topk_idx = np.argpartition(S[:, j], -k)[-k:]
        W[topk_idx, j] = S[topk_idx, j]
    return W.astype(np.float32)

def knn_recommend(X_csr, W, k=10):
    scores = X_csr.toarray() @ W
    seen = X_csr.toarray() > 0
    scores[seen] = -np.inf
    topk = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    return topk

def knn_score(X_csr, W):
    return X_csr.toarray() @ W

# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────

def run_implicit_model_benchmark(model, X_csr, algo_name, scale_name):
    """Benchmark recommend/score for implicit library models."""
    results = []
    n_users, n_items = X_csr.shape

    # recommend — implicit uses (userid, user_items) per user, or batch recommend
    # Use the batch recommend API
    def do_recommend(k_val):
        def _inner():
            ids, scores = model.recommend(
                np.arange(n_users), X_csr, N=k_val, filter_already_liked_items=True
            )
            return ids
        return _inner

    for k_val in [10, 50]:
        t, mem = bench(do_recommend(k_val))
        results.append({
            "scale": scale_name, "algorithm": algo_name,
            "operation": f"recommend(k={k_val})",
            "time_seconds": t, "memory_bytes": mem,
            "n_users": n_users, "n_items": n_items,
            "throughput": n_users / t,
        })

    # score — not all implicit models expose score well; skip if dense > 4 GiB
    if n_users * n_items * 4 < 4 * 1024**3:
        def do_score():
            # Reconstruct full score matrix from factors
            scores = model.user_factors @ model.item_factors.T
            return scores
        t, mem = bench(do_score)
        results.append({
            "scale": scale_name, "algorithm": algo_name,
            "operation": "score(full)",
            "time_seconds": t, "memory_bytes": mem,
            "n_users": n_users, "n_items": n_items,
            "throughput": n_users / t,
        })
    else:
        results.append({
            "scale": scale_name, "algorithm": algo_name,
            "operation": "score(full)",
            "time_seconds": float("nan"), "memory_bytes": 0,
            "n_users": n_users, "n_items": n_items,
            "throughput": float("nan"),
        })

    # similar_items — batch query
    def do_similar():
        n_q = min(1000, n_items)
        ids, scores = model.similar_items(np.arange(n_q), N=10)
        return ids
    t, mem = bench(do_similar)
    n_q = min(1000, n_items)
    results.append({
        "scale": scale_name, "algorithm": algo_name,
        "operation": f"similar_items(×{n_q})",
        "time_seconds": t, "memory_bytes": mem,
        "n_users": n_users, "n_items": n_items,
        "throughput": n_q / t,
    })

    return results

def run_numpy_model_benchmark(X_csr, W, algo_name, scale_name,
                              recommend_fn, score_fn):
    """Benchmark recommend/score for numpy-based models (EASE, ItemKNN)."""
    results = []
    n_users, n_items = X_csr.shape

    for k_val in [10, 50]:
        t, mem = bench(lambda k=k_val: recommend_fn(X_csr, W, k=k))
        results.append({
            "scale": scale_name, "algorithm": algo_name,
            "operation": f"recommend(k={k_val})",
            "time_seconds": t, "memory_bytes": mem,
            "n_users": n_users, "n_items": n_items,
            "throughput": n_users / t,
        })

    if n_users * n_items * 4 < 4 * 1024**3:
        t, mem = bench(lambda: score_fn(X_csr, W))
        results.append({
            "scale": scale_name, "algorithm": algo_name,
            "operation": "score(full)",
            "time_seconds": t, "memory_bytes": mem,
            "n_users": n_users, "n_items": n_items,
            "throughput": n_users / t,
        })
    else:
        results.append({
            "scale": scale_name, "algorithm": algo_name,
            "operation": "score(full)",
            "time_seconds": float("nan"), "memory_bytes": 0,
            "n_users": n_users, "n_items": n_items,
            "throughput": float("nan"),
        })

    return results

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 78)
    print("Python (implicit + sklearn) — Inference Benchmark")
    print("=" * 78)
    print(f"  implicit:  {implicit.__version__}")
    print(f"  numpy:     {np.__version__}")
    print(f"  CPU count: {os.cpu_count()}")
    print()

    all_results = []

    for scale in SCALES:
        print("━" * 78)
        print(f"Scale: {scale['name']} ({scale['n_users']} × {scale['n_items']}, "
              f"density={scale['density']:.3f})")
        print("━" * 78)

        X = generate_matrix(scale["n_users"], scale["n_items"], scale["density"])
        print(f"  Matrix: {X.shape[0]} users × {X.shape[1]} items, nnz={X.nnz}\n")

        # ── implicit models ──
        for algo_name, model_fn in mf_models():
            try:
                model = model_fn()
            except Exception as e:
                print(f"  {algo_name:<12}  SKIPPED ({e})")
                continue

            print(f"  {algo_name:<12}  training... ", end="", flush=True)
            gc.collect()
            t0 = time.perf_counter()
            model.fit(X)
            t_train = time.perf_counter() - t0
            print(f"{t_train:.2f}s  →  benchmarking inference...")

            results = run_implicit_model_benchmark(model, X, algo_name, scale["name"])
            all_results.extend(results)

            for r in results:
                if np.isnan(r["time_seconds"]):
                    print(f"    {r['operation']:<25}  {'SKIPPED':>8}  (>4 GiB output)")
                else:
                    print(f"    {r['operation']:<25}  {r['time_seconds']:8.4f} s  "
                          f"{fmt_bytes(r['memory_bytes']):>10}  "
                          f"{r['throughput']:10.0f} /s")
            print()

        # ── EASE (numpy) — skip for large ──
        if scale["n_items"] <= 5_000:
            print(f"  {'EASE-np':<12}  training... ", end="", flush=True)
            gc.collect()
            t0 = time.perf_counter()
            B = ease_fit(X, lam=500.0)
            t_train = time.perf_counter() - t0
            print(f"{t_train:.2f}s  →  benchmarking inference...")

            results = run_numpy_model_benchmark(
                X, B, "EASE-np", scale["name"], ease_recommend, ease_score)
            all_results.extend(results)
            for r in results:
                if np.isnan(r["time_seconds"]):
                    print(f"    {r['operation']:<25}  {'SKIPPED':>8}  (>4 GiB)")
                else:
                    print(f"    {r['operation']:<25}  {r['time_seconds']:8.4f} s  "
                          f"{fmt_bytes(r['memory_bytes']):>10}  "
                          f"{r['throughput']:10.0f} /s")
            print()
        else:
            print(f"  {'EASE-np':<12}  SKIPPED (too expensive at this scale)")

        # ── ItemKNN (sklearn) — skip for large ──
        if scale["n_items"] <= 5_000:
            print(f"  {'ItemKNN-sk':<12}  training... ", end="", flush=True)
            gc.collect()
            t0 = time.perf_counter()
            W_knn = knn_fit(X, k=50)
            t_train = time.perf_counter() - t0
            print(f"{t_train:.2f}s  →  benchmarking inference...")

            results = run_numpy_model_benchmark(
                X, W_knn, "ItemKNN-sk", scale["name"], knn_recommend, knn_score)
            all_results.extend(results)
            for r in results:
                if np.isnan(r["time_seconds"]):
                    print(f"    {r['operation']:<25}  {'SKIPPED':>8}  (>4 GiB)")
                else:
                    print(f"    {r['operation']:<25}  {r['time_seconds']:8.4f} s  "
                          f"{fmt_bytes(r['memory_bytes']):>10}  "
                          f"{r['throughput']:10.0f} /s")
            print()
        else:
            print(f"  {'ItemKNN-sk':<12}  SKIPPED (too expensive at this scale)")

    # ── Save CSV ──
    outpath = os.path.join(os.path.dirname(__file__), "results_recommend_python.csv")
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["scale", "algorithm", "operation", "time_seconds",
                      "memory_bytes", "n_users", "n_items", "throughput"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in r.items()})

    # ── Summary ──
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Scale':<8} {'Algorithm':<12} {'Operation':<25} {'Time (s)':>10} "
          f"{'Memory':>10} {'Throughput':>12}")
    print("─" * 78)
    for r in all_results:
        if np.isnan(r["time_seconds"]):
            print(f"{r['scale']:<8} {r['algorithm']:<12} {r['operation']:<25} "
                  f"{'SKIP':>10} {'-':>10} {'-':>12}")
        else:
            print(f"{r['scale']:<8} {r['algorithm']:<12} {r['operation']:<25} "
                  f"{r['time_seconds']:>10.4f} {fmt_bytes(r['memory_bytes']):>10} "
                  f"{r['throughput']:>10.0f} /s")
    print("─" * 78)
    print(f"\nResults saved to: {outpath}")


if __name__ == "__main__":
    main()
