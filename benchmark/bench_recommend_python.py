#!/usr/bin/env python3
"""
Benchmark recommend / score / predict throughput and memory for Python libraries.

Counterpart to bench_recommend.jl — same data sizes, same algorithms where possible.

Run:
    python benchmark/bench_recommend_python.py

Libraries:
    - implicit (ALS, BPR, LMF) — GPU disabled, CPU only
    - sklearn (NearestNeighbors for ItemKNN-like)
    - scipy + numpy (for EASE, dense scoring)

Memory is tracked via tracemalloc (peak allocation) and RSS from /proc.
"""

import csv
import gc
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# ─────────────────────────────────────────────
# Memory monitoring
# ─────────────────────────────────────────────

def rss_bytes():
    """Current RSS from /proc/self/statm (Linux)."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * page_size

def peak_rss_bytes():
    """Peak RSS (VmHWM) from /proc/self/status."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return 0

def fmt_bytes(b):
    if b < 0:
        return "0 B"
    if b < 1024:
        return f"{b} B"
    if b < 1024**2:
        return f"{b/1024:.1f} KiB"
    if b < 1024**3:
        return f"{b/1024**2:.1f} MiB"
    return f"{b/1024**3:.1f} GiB"


def bench(f, warmup=True, repeats=3):
    """Run f() `repeats` times, return (median_time, median_rss_delta)."""
    if warmup:
        f()  # warmup
    gc.collect()

    times = []
    mem_deltas = []
    for _ in range(repeats):
        gc.collect()
        rss_before = rss_bytes()
        t0 = time.perf_counter()
        result = f()
        t1 = time.perf_counter()
        rss_after = rss_bytes()
        times.append(t1 - t0)
        mem_deltas.append(max(0, rss_after - rss_before))
        del result
        gc.collect()

    times.sort()
    mem_deltas.sort()
    median_t = times[len(times) // 2]
    median_m = mem_deltas[len(mem_deltas) // 2]
    return median_t, median_m


# ─────────────────────────────────────────────
# Synthetic data generators (Float32)
# ─────────────────────────────────────────────

def generate_matrix(n_users, n_items, density, seed=42):
    """Generate random sparse user-item interaction matrix (CSR, Float32)."""
    rng = np.random.default_rng(seed)
    nnz_target = int(n_users * n_items * density)
    rows = rng.integers(0, n_users, size=nnz_target)
    cols = rng.integers(0, n_items, size=nnz_target)
    vals = np.ones(nnz_target, dtype=np.float32)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    X.sum_duplicates()
    return X


# ─────────────────────────────────────────────
# Result storage
# ─────────────────────────────────────────────

@dataclass
class InferenceResult:
    scale: str
    algorithm: str
    operation: str
    time_seconds: float
    memory_bytes: int
    n_users: int
    n_items: int
    throughput: float


# ─────────────────────────────────────────────
# Benchmark configurations (same as Julia)
# ─────────────────────────────────────────────

SCALES = [
    {"name": "small",  "n_users": 1_000,  "n_items": 500,   "density": 0.05},
    {"name": "medium", "n_users": 10_000, "n_items": 2_000, "density": 0.02},
    {"name": "large",  "n_users": 50_000, "n_items": 5_000, "density": 0.005},
]


# ─────────────────────────────────────────────
# Model runners
# ─────────────────────────────────────────────

def run_implicit_benchmark(model_class, model_kwargs, algo_name, X_csr, scale_name):
    """Train implicit model, benchmark recommend and score."""
    results = []
    n_users, n_items = X_csr.shape

    # implicit 0.7+ expects user-item matrix for fit()
    model = model_class(**model_kwargs)
    print(f"  {algo_name:<12}  training... ", end="", flush=True)
    gc.collect()
    t0 = time.perf_counter()
    model.fit(X_csr)
    t_train = time.perf_counter() - t0
    print(f"{t_train:.2f}s  →  benchmarking inference...")

    userids = np.arange(n_users)

    # recommend for all users (top-10)
    def recommend_k10():
        ids, scores = model.recommend(userids, X_csr[userids], N=10,
                                      filter_already_liked_items=False)
        return ids

    t, m = bench(recommend_k10)
    results.append(InferenceResult(scale_name, algo_name, "recommend(k=10)",
                                   t, m, n_users, n_items, n_users / t))

    # recommend top-50
    def recommend_k50():
        ids, scores = model.recommend(userids, X_csr[userids], N=50,
                                      filter_already_liked_items=False)
        return ids

    t, m = bench(recommend_k50)
    results.append(InferenceResult(scale_name, algo_name, "recommend(k=50)",
                                   t, m, n_users, n_items, n_users / t))

    # Full score matrix — only if < 2 GiB
    score_size = n_users * n_items * 4
    if score_size < 2 * 1024**3:
        if hasattr(model, 'user_factors') and hasattr(model, 'item_factors'):
            def score_full():
                return model.user_factors @ model.item_factors.T

            t, m = bench(score_full)
            results.append(InferenceResult(scale_name, algo_name, "score(full)",
                                           t, m, n_users, n_items, n_users / t))
        else:
            results.append(InferenceResult(scale_name, algo_name, "score(full)",
                                           float('nan'), 0, n_users, n_items, float('nan')))
    else:
        results.append(InferenceResult(scale_name, algo_name, "score(full)",
                                       float('nan'), 0, n_users, n_items, float('nan')))

    # similar_items
    if hasattr(model, 'similar_items'):
        n_queries = min(1000, n_items)
        def sim_items():
            for i in range(n_queries):
                model.similar_items(i, N=10)

        t, m = bench(sim_items)
        results.append(InferenceResult(scale_name, algo_name,
                                       f"similar_items(×{n_queries})",
                                       t, m, n_users, n_items, n_queries / t))

    for r in results:
        if np.isnan(r.time_seconds):
            print(f"    {r.operation:<25}  SKIPPED (>2 GiB output)")
        else:
            print(f"    {r.operation:<25}  {r.time_seconds:8.4f} s  "
                  f"{fmt_bytes(r.memory_bytes):>10}  {r.throughput:10.0f} users/s")
    print()

    del model
    gc.collect()
    return results


def run_ease_benchmark(X_csr, scale_name, lam=500.0):
    """EASE^R: closed-form B = I - P·diag(1/diag(P)) where P = (G + λI)⁻¹."""
    n_users, n_items = X_csr.shape
    results = []

    print(f"  {'EASE':<12}  training... ", end="", flush=True)
    gc.collect()
    t0 = time.perf_counter()

    # Gram matrix
    X = X_csr.toarray().astype(np.float32)
    G = X.T @ X
    G += lam * np.eye(n_items, dtype=np.float32)
    P = np.linalg.inv(G)
    B = P / (-np.diag(P)[np.newaxis, :])
    np.fill_diagonal(B, 0.0)

    t_train = time.perf_counter() - t0
    print(f"{t_train:.2f}s  →  benchmarking inference...")

    # recommend(k=10): scores = X @ B, then top-k
    def recommend_k10():
        scores = X_csr @ B
        return np.argpartition(-scores, 10, axis=1)[:, :10]

    t, m = bench(recommend_k10)
    results.append(InferenceResult(scale_name, "EASE", "recommend(k=10)",
                                   t, m, n_users, n_items, n_users / t))

    def recommend_k50():
        scores = X_csr @ B
        return np.argpartition(-scores, 50, axis=1)[:, :50]

    t, m = bench(recommend_k50)
    results.append(InferenceResult(scale_name, "EASE", "recommend(k=50)",
                                   t, m, n_users, n_items, n_users / t))

    # score
    score_size = n_users * n_items * 4
    if score_size < 2 * 1024**3:
        def score_full():
            return X_csr @ B

        t, m = bench(score_full)
        results.append(InferenceResult(scale_name, "EASE", "score(full)",
                                       t, m, n_users, n_items, n_users / t))
    else:
        results.append(InferenceResult(scale_name, "EASE", "score(full)",
                                       float('nan'), 0, n_users, n_items, float('nan')))

    for r in results:
        if np.isnan(r.time_seconds):
            print(f"    {r.operation:<25}  SKIPPED (>2 GiB output)")
        else:
            print(f"    {r.operation:<25}  {r.time_seconds:8.4f} s  "
                  f"{fmt_bytes(r.memory_bytes):>10}  {r.throughput:10.0f} users/s")
    print()

    del B, G, P, X
    gc.collect()
    return results


def run_knn_benchmark(X_csr, scale_name, k=50):
    """Item-based KNN using sklearn cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    n_users, n_items = X_csr.shape
    results = []

    print(f"  {'ItemKNN':<12}  training... ", end="", flush=True)
    gc.collect()
    t0 = time.perf_counter()

    # Compute item-item cosine similarity, keep top-k per column
    sim = cosine_similarity(X_csr.T, dense_output=False)
    # Zero diagonal
    sim.setdiag(0)
    sim.eliminate_zeros()

    # Keep top-k per item (column)
    sim_csc = sim.tocsc()
    for j in range(n_items):
        start, end_ = sim_csc.indptr[j], sim_csc.indptr[j + 1]
        if end_ - start > k:
            data = sim_csc.data[start:end_]
            threshold = np.partition(data, -k)[-k]
            mask = data < threshold
            sim_csc.data[start:end_][mask] = 0
    sim_csc.eliminate_zeros()
    W = sim_csc.astype(np.float32)

    t_train = time.perf_counter() - t0
    print(f"{t_train:.2f}s  →  benchmarking inference...")

    # recommend(k=10): scores = X @ W, then top-k
    def recommend_k10():
        scores = X_csr @ W
        if sp.issparse(scores):
            scores = scores.toarray()
        return np.argpartition(-scores, 10, axis=1)[:, :10]

    t, m = bench(recommend_k10)
    results.append(InferenceResult(scale_name, "ItemKNN", "recommend(k=10)",
                                   t, m, n_users, n_items, n_users / t))

    def recommend_k50():
        scores = X_csr @ W
        if sp.issparse(scores):
            scores = scores.toarray()
        return np.argpartition(-scores, 50, axis=1)[:, :50]

    t, m = bench(recommend_k50)
    results.append(InferenceResult(scale_name, "ItemKNN", "recommend(k=50)",
                                   t, m, n_users, n_items, n_users / t))

    # score
    score_size = n_users * n_items * 4
    if score_size < 2 * 1024**3:
        def score_full():
            s = X_csr @ W
            return s.toarray() if sp.issparse(s) else s

        t, m = bench(score_full)
        results.append(InferenceResult(scale_name, "ItemKNN", "score(full)",
                                       t, m, n_users, n_items, n_users / t))
    else:
        results.append(InferenceResult(scale_name, "ItemKNN", "score(full)",
                                       float('nan'), 0, n_users, n_items, float('nan')))

    for r in results:
        if np.isnan(r.time_seconds):
            print(f"    {r.operation:<25}  SKIPPED (>2 GiB output)")
        else:
            print(f"    {r.operation:<25}  {r.time_seconds:8.4f} s  "
                  f"{fmt_bytes(r.memory_bytes):>10}  {r.throughput:10.0f} users/s")
    print()

    del W, sim, sim_csc
    gc.collect()
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import platform
    print("=" * 78)
    print("Python — Inference (recommend / score) Benchmark")
    print("=" * 78)
    print(f"  Python:       {platform.python_version()}")
    print(f"  NumPy:        {np.__version__}")
    print(f"  Peak RSS:     {fmt_bytes(peak_rss_bytes())}")
    print()

    all_results = []

    for scale in SCALES:
        name = scale["name"]
        n_users = scale["n_users"]
        n_items = scale["n_items"]
        density = scale["density"]

        print("━" * 78)
        print(f"Scale: {name} ({n_users} users × {n_items} items, density={density:.3f})")
        print("━" * 78)

        X = generate_matrix(n_users, n_items, density)
        nnz = X.nnz
        sparse_mb = (nnz * (4 + 4) + (n_users + 1) * 4) / 1024**2
        print(f"  Matrix: {n_users} users × {n_items} items, nnz={nnz} "
              f"({sparse_mb:.1f} MiB sparse)\n")

        # implicit models
        from implicit.cpu.als import AlternatingLeastSquares
        from implicit.cpu.bpr import BayesianPersonalizedRanking
        from implicit.cpu.lmf import LogisticMatrixFactorization

        implicit_models = [
            (AlternatingLeastSquares,
             {"factors": 64, "regularization": 0.1, "iterations": 5, "num_threads": 4},
             "ALS(implicit)"),
            (BayesianPersonalizedRanking,
             {"factors": 64, "regularization": 0.01, "iterations": 5, "num_threads": 4},
             "BPR(implicit)"),
            (LogisticMatrixFactorization,
             {"factors": 64, "regularization": 0.6, "iterations": 5, "num_threads": 4},
             "LMF(implicit)"),
        ]

        for model_class, kwargs, algo_name in implicit_models:
            results = run_implicit_benchmark(
                model_class, kwargs, algo_name, X, name)
            all_results.extend(results)

        # EASE
        all_results.extend(run_ease_benchmark(X, name))

        # ItemKNN
        all_results.extend(run_knn_benchmark(X, name))

        del X
        gc.collect()

    # ── Save CSV ──
    outpath = os.path.join(os.path.dirname(__file__), "results_recommend_python.csv")
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scale", "algorithm", "operation", "time_seconds",
                         "memory_bytes", "n_users", "n_items", "throughput"])
        for r in all_results:
            writer.writerow([r.scale, r.algorithm, r.operation,
                             f"{r.time_seconds:.6f}", r.memory_bytes,
                             r.n_users, r.n_items, f"{r.throughput:.1f}"])

    # ── Summary ──
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Peak RSS: {fmt_bytes(peak_rss_bytes())}")
    print()
    print(f"{'Scale':<8} {'Algorithm':<14} {'Operation':<25} "
          f"{'Time (s)':>10} {'Memory':>10} {'Throughput':>12}")
    print("─" * 78)
    for r in all_results:
        if np.isnan(r.time_seconds):
            print(f"{r.scale:<8} {r.algorithm:<14} {r.operation:<25} "
                  f"{'SKIP':>10} {'-':>10} {'-':>12}")
        else:
            print(f"{r.scale:<8} {r.algorithm:<14} {r.operation:<25} "
                  f"{r.time_seconds:10.4f} {fmt_bytes(r.memory_bytes):>10} "
                  f"{r.throughput:10.0f} /s")
    print("─" * 78)
    print(f"\nResults saved to: {outpath}")


if __name__ == "__main__":
    main()
