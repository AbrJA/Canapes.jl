#!/usr/bin/env python3
"""
Generate Python reference fixtures for Gideon validation.

Usage:
    python3 validation/fixtures_py.py

Outputs:
    /tmp/gideon_fixtures/python/
    - X_small.csv
    - X_small_dims.csv
    - X_train.csv
    - X_train_dims.csv
    - X_test.csv
    - X_test_dims.csv
    - py_als_scores.csv
    - py_ials_scores.csv
    - py_eals_scores.csv
    - py_bpr_scores.csv
        - py_lmf_scores.csv (optional)
        - py_ease_B.csv
        - py_slim_W.csv (optional)
        - py_softimpute_recon.csv
        - py_softimpute_svals.csv
    - py_ials_metrics.json
    - py_eals_metrics.json
    - py_bpr_metrics.json
    - py_lmf_metrics.json (optional)
    - py_slim_metrics.json (optional)
    - meta.json
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
from scipy.sparse import csr_matrix


def write_triplet(x: csr_matrix, path: pathlib.Path) -> None:
    coo = x.tocoo()
    with path.open("w", encoding="utf-8") as f:
        f.write("row,col,val\n")
        for r, c, v in zip(coo.row, coo.col, coo.data):
            f.write(f"{r + 1},{c + 1},{float(v)}\n")


def ndcg_at_k(preds: np.ndarray, test: csr_matrix, k: int = 10) -> float:
    vals = []
    for u in range(preds.shape[0]):
        relevant = set(test[u].indices.tolist())
        if not relevant:
            continue
        dcg = 0.0
        for r, item in enumerate(preds[u, :k]):
            if int(item) in relevant:
                dcg += 1.0 / np.log2(r + 2.0)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def recall_at_k(preds: np.ndarray, test: csr_matrix, k: int = 10) -> float:
    vals = []
    for u in range(preds.shape[0]):
        relevant = set(test[u].indices.tolist())
        if not relevant:
            continue
        hits = sum(1 for item in preds[u, :k] if int(item) in relevant)
        vals.append(hits / len(relevant))
    return float(np.mean(vals)) if vals else 0.0


def split_train_test(x: csr_matrix, rng: np.random.Generator, test_frac: float = 0.2):
    x = x.tocsr()
    n_users, n_items = x.shape
    train_rows: list[int] = []
    train_cols: list[int] = []
    train_vals: list[float] = []
    test_rows: list[int] = []
    test_cols: list[int] = []
    test_vals: list[float] = []

    for u in range(n_users):
        start, end = x.indptr[u], x.indptr[u + 1]
        cols_u = x.indices[start:end]
        vals_u = x.data[start:end]
        nnz_u = len(cols_u)
        if nnz_u == 0:
            continue

        n_test = max(1, int(np.floor(nnz_u * test_frac))) if nnz_u > 1 else 0
        perm = rng.permutation(nnz_u)
        test_idx = set(perm[:n_test].tolist())

        for j in range(nnz_u):
            c = int(cols_u[j])
            v = float(vals_u[j])
            if j in test_idx:
                test_rows.append(u)
                test_cols.append(c)
                test_vals.append(v)
            else:
                train_rows.append(u)
                train_cols.append(c)
                train_vals.append(v)

    train = csr_matrix((np.array(train_vals, dtype=np.float32),
                        (np.array(train_rows, dtype=np.int32), np.array(train_cols, dtype=np.int32))),
                       shape=(n_users, n_items))
    test = csr_matrix((np.array(test_vals, dtype=np.float32),
                       (np.array(test_rows, dtype=np.int32), np.array(test_cols, dtype=np.int32))),
                      shape=(n_users, n_items))
    train.sum_duplicates()
    test.sum_duplicates()
    return train, test


def topk_from_scores(scores: np.ndarray, train: csr_matrix | None = None, k: int = 10) -> np.ndarray:
    s = scores.copy()
    if train is not None:
        tr = train.tocsr()
        for u in range(tr.shape[0]):
            start, end = tr.indptr[u], tr.indptr[u + 1]
            if start < end:
                s[u, tr.indices[start:end]] = -np.inf
    part = np.argpartition(-s, kth=k - 1, axis=1)[:, :k]
    row = np.arange(s.shape[0])[:, None]
    order = np.argsort(-s[row, part], axis=1)
    return part[row, order]


def soft_impute_dense(
    train: csr_matrix,
    rank: int = 10,
    lam: float = 0.1,
    max_iter: int = 40,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    x = train.toarray().astype(np.float64)
    mask = x != 0
    m, n = x.shape
    z = x.copy()
    prev = z.copy()
    max_rank = min(rank, m, n)

    for _ in range(max_iter):
        u, s, vt = np.linalg.svd(z, full_matrices=False)
        s = np.maximum(s[:max_rank] - lam, 0.0)
        r = int(np.sum(s > 0))
        if r == 0:
            z_new = np.zeros_like(z)
            s_out = s
        else:
            u_r = u[:, :r]
            vt_r = vt[:r, :]
            z_new = (u_r * s[:r]) @ vt_r
            s_out = s[:r]
        z_new[mask] = x[mask]

        num = np.linalg.norm(z_new - prev)
        den = np.linalg.norm(prev) + 1e-12
        if num / den < tol:
            return z_new, s_out
        prev = z_new
        z = z_new

    return z, s_out


def try_fit_slim(train: csr_matrix, l1: float = 0.01, l2: float = 0.1, max_iter: int = 200):
    try:
        from sklearn.linear_model import ElasticNet
    except Exception:
        return None

    x = train.tocsc()
    _, n_items = x.shape
    w = np.zeros((n_items, n_items), dtype=np.float64)
    alpha = l1 + l2
    if alpha <= 0:
        alpha = 1e-6
    l1_ratio = l1 / alpha

    for j in range(n_items):
        y = x[:, j].toarray().ravel()
        x_j = x.copy().tolil()
        x_j[:, j] = 0.0
        x_j = x_j.tocsr()
        reg = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            positive=True,
            max_iter=max_iter,
            selection="cyclic",
            tol=1e-4,
        )
        reg.fit(x_j, y)
        w[:, j] = reg.coef_
        w[j, j] = 0.0
    return w


def main() -> int:
    try:
        import implicit
    except Exception as exc:  # pragma: no cover
        print("[ERROR] Missing Python dependency: implicit")
        print("Install with: pip install implicit")
        print(f"Details: {exc}")
        return 2

    out_dir = pathlib.Path(
        os.environ.get("GIDEON_PY_FIXTURE_DIR", "/tmp/gideon_fixtures/python")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    rng = np.random.default_rng(seed)

    n_users = 120
    n_items = 100
    density = 0.06

    nnz = int(n_users * n_items * density)
    rows = rng.integers(0, n_users, size=nnz)
    cols = rng.integers(0, n_items, size=nnz)
    vals = rng.integers(1, 6, size=nnz).astype(np.float32)
    x = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    x.sum_duplicates()

    rank = 16
    als_iters = 20
    bpr_iters = 40

    train, test = split_train_test(x, rng, test_frac=0.2)

    als = implicit.als.AlternatingLeastSquares(
        factors=rank,
        regularization=0.1,
        alpha=40.0,
        iterations=als_iters,
        random_state=seed,
        use_gpu=False,
    )
    als.fit(x)

    ials = implicit.als.AlternatingLeastSquares(
        factors=rank,
        regularization=0.01,
        alpha=40.0,
        iterations=15,
        random_state=seed,
        use_gpu=False,
    )
    ials.fit(x)

    # EALS surrogate (no native EALS in implicit): lower confidence weighting.
    eals = implicit.als.AlternatingLeastSquares(
        factors=rank,
        regularization=0.01,
        alpha=10.0,
        iterations=10,
        random_state=seed,
        use_gpu=False,
    )
    eals.fit(x)

    bpr_cls = None
    if hasattr(implicit.bpr, "BPR"):
        bpr_cls = implicit.bpr.BPR
    elif hasattr(implicit.bpr, "BayesianPersonalizedRanking"):
        bpr_cls = implicit.bpr.BayesianPersonalizedRanking
    else:
        raise RuntimeError("No compatible BPR class found in implicit.bpr")

    bpr = bpr_cls(
        factors=rank,
        regularization=0.01,
        learning_rate=0.05,
        iterations=bpr_iters,
        random_state=seed,
        use_gpu=False,
    )
    bpr.fit(x)

    als_scores = als.user_factors @ als.item_factors.T
    ials_scores = ials.user_factors @ ials.item_factors.T
    eals_scores = eals.user_factors @ eals.item_factors.T
    bpr_scores = bpr.user_factors @ bpr.item_factors.T

    # Optional Logistic Matrix Factorization parity fixture.
    lmf_scores = None
    lmf_class = None
    try:
        lmf_mod = getattr(implicit, "lmf", None)
        if lmf_mod is not None and hasattr(lmf_mod, "LogisticMatrixFactorization"):
            lmf_class = lmf_mod.LogisticMatrixFactorization
            lmf = lmf_class(
                factors=rank,
                learning_rate=1.0,
                regularization=0.6,
                iterations=30,
                random_state=seed,
            )
            lmf.fit(x)
            lmf_scores = lmf.user_factors @ lmf.item_factors.T
    except Exception:
        # Keep fixture generation resilient; validate script will skip LogisticMF if absent.
        lmf_scores = None

    # Deterministic Python EASE reference implementation.
    ease_lambda = 100.0
    gram = (x.T @ x).toarray().astype(np.float64)
    n_items = gram.shape[0]
    gram[np.diag_indices(n_items)] += ease_lambda
    p = np.linalg.inv(gram)
    b = -p / np.diag(p)
    b[np.diag_indices(n_items)] = 0.0

    write_triplet(x, out_dir / "X_small.csv")
    with (out_dir / "X_small_dims.csv").open("w", encoding="utf-8") as f:
        f.write("nr,nc\n")
        f.write(f"{n_users},{n_items}\n")

    np.savetxt(out_dir / "py_als_scores.csv", als_scores, delimiter=",")
    np.savetxt(out_dir / "py_ials_scores.csv", ials_scores, delimiter=",")
    np.savetxt(out_dir / "py_eals_scores.csv", eals_scores, delimiter=",")
    np.savetxt(out_dir / "py_bpr_scores.csv", bpr_scores, delimiter=",")
    np.savetxt(out_dir / "py_ease_B.csv", b, delimiter=",")
    if lmf_scores is not None:
        np.savetxt(out_dir / "py_lmf_scores.csv", lmf_scores, delimiter=",")

    # Optional SLIM reference using sklearn ElasticNet.
    slim_l1 = 0.001
    slim_l2 = 0.01
    slim_w = try_fit_slim(train, l1=slim_l1, l2=slim_l2, max_iter=300)
    if slim_w is not None:
        np.savetxt(out_dir / "py_slim_W.csv", slim_w, delimiter=",")

    # SoftImpute reference (dense iterative soft-threshold SVD).
    soft_recon, soft_svals = soft_impute_dense(train, rank=10, lam=0.1, max_iter=40, tol=1e-4)
    np.savetxt(out_dir / "py_softimpute_recon.csv", soft_recon, delimiter=",")
    np.savetxt(out_dir / "py_softimpute_svals.csv", soft_svals[None, :], delimiter=",")

    write_triplet(train, out_dir / "X_train.csv")
    with (out_dir / "X_train_dims.csv").open("w", encoding="utf-8") as f:
        f.write("nr,nc\n")
        f.write(f"{n_users},{n_items}\n")
    write_triplet(test, out_dir / "X_test.csv")
    with (out_dir / "X_test_dims.csv").open("w", encoding="utf-8") as f:
        f.write("nr,nc\n")
        f.write(f"{n_users},{n_items}\n")

    ials_preds = topk_from_scores(ials_scores, train=train, k=10)
    ials_metrics = {
        "k": 10,
        "ndcg": ndcg_at_k(ials_preds, test, k=10),
        "recall": recall_at_k(ials_preds, test, k=10),
    }
    (out_dir / "py_ials_metrics.json").write_text(json.dumps(ials_metrics, indent=2), encoding="utf-8")

    eals_preds = topk_from_scores(eals_scores, train=train, k=10)
    eals_metrics = {
        "k": 10,
        "ndcg": ndcg_at_k(eals_preds, test, k=10),
        "recall": recall_at_k(eals_preds, test, k=10),
    }
    (out_dir / "py_eals_metrics.json").write_text(json.dumps(eals_metrics, indent=2), encoding="utf-8")

    bpr_preds = topk_from_scores(bpr_scores, train=train, k=10)
    bpr_metrics = {
        "k": 10,
        "ndcg": ndcg_at_k(bpr_preds, test, k=10),
        "recall": recall_at_k(bpr_preds, test, k=10),
    }
    (out_dir / "py_bpr_metrics.json").write_text(json.dumps(bpr_metrics, indent=2), encoding="utf-8")

    if lmf_scores is not None:
        lmf_preds = topk_from_scores(lmf_scores, train=train, k=10)
        lmf_metrics = {
            "k": 10,
            "ndcg": ndcg_at_k(lmf_preds, test, k=10),
            "recall": recall_at_k(lmf_preds, test, k=10),
        }
        (out_dir / "py_lmf_metrics.json").write_text(json.dumps(lmf_metrics, indent=2), encoding="utf-8")

    if slim_w is not None:
        slim_scores = train.toarray().astype(np.float64) @ slim_w
        slim_preds = topk_from_scores(slim_scores, train=train, k=10)
        slim_metrics = {
            "k": 10,
            "ndcg": ndcg_at_k(slim_preds, test, k=10),
            "recall": recall_at_k(slim_preds, test, k=10),
        }
        (out_dir / "py_slim_metrics.json").write_text(json.dumps(slim_metrics, indent=2), encoding="utf-8")

    # ── PureSVD reference (scipy truncated SVD) ──────────────────────────
    print("[PureSVD] Computing scipy.sparse.linalg.svds reference...")
    from scipy.sparse.linalg import svds as scipy_svds
    pure_rank = 10
    u_svd, s_svd, vt_svd = scipy_svds(train.astype(np.float64), k=pure_rank)
    # scipy returns ascending order; flip to descending
    idx = np.argsort(-s_svd)
    u_svd = u_svd[:, idx]
    s_svd = s_svd[idx]
    vt_svd = vt_svd[idx, :]
    recon_svd = (u_svd * s_svd) @ vt_svd
    np.savetxt(out_dir / "py_puresvd_svals.csv", s_svd[None, :], delimiter=",")
    np.savetxt(out_dir / "py_puresvd_recon.csv", recon_svd, delimiter=",")
    print(f"  singular values: {s_svd[:5]}")

    # ── ItemKNN cosine reference (scipy/sklearn, optional) ────────────────
    knn_k = 20
    print("[ItemKNN] Computing cosine KNN reference...")
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        print("[ItemKNN] Skipped: scikit-learn unavailable")
        knn_metrics = None
    else:
        x_dense = x.toarray().astype(np.float64)
        # Item-item cosine similarity from X^T X normalized
        sim = cosine_similarity(x_dense.T)
        np.fill_diagonal(sim, 0.0)
        # Keep top-k per column (column j: top-k most similar to item j)
        sim_topk = np.zeros_like(sim)
        for j in range(sim.shape[1]):
            topk_idx = np.argpartition(-sim[:, j], knn_k)[:knn_k]
            sim_topk[topk_idx, j] = sim[topk_idx, j]
        # Row-normalize
        row_sums = sim_topk.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sim_topk_norm = sim_topk / row_sums
        knn_scores = x_dense @ sim_topk_norm
        np.savetxt(out_dir / "py_knn_W.csv", sim_topk_norm, delimiter=",")
        np.savetxt(out_dir / "py_knn_scores.csv", knn_scores, delimiter=",")
        knn_preds = topk_from_scores(knn_scores, train=train, k=10)
        knn_metrics = {
            "k": 10,
            "knn_k": knn_k,
            "ndcg": ndcg_at_k(knn_preds, test, k=10),
            "recall": recall_at_k(knn_preds, test, k=10),
        }
        (out_dir / "py_knn_metrics.json").write_text(json.dumps(knn_metrics, indent=2), encoding="utf-8")
        print(f"  W nnz: {np.count_nonzero(sim_topk_norm)}, NDCG@10: {knn_metrics['ndcg']:.4f}")

    # ── ADMMSLIM reference (closed-form ADMM, pure numpy) ──────────────────
    print("[ADMMSLIM] Computing ADMM-SLIM reference...")
    x_dense = x.toarray().astype(np.float64)
    admm_l1 = 0.01
    admm_l2 = 100.0
    admm_rho = 1.0
    admm_max_iter = 100
    admm_tol = 1e-5
    G = x_dense.T @ x_dense  # Gram
    n_it = G.shape[0]
    lhs = G + (admm_l2 + admm_rho) * np.eye(n_it)
    lhs_inv = np.linalg.inv(lhs)  # pre-factor
    B = np.zeros((n_it, n_it))
    Z = np.zeros((n_it, n_it))
    U = np.zeros((n_it, n_it))
    for it in range(admm_max_iter):
        rhs_admm = G + admm_rho * (Z - U)
        B = lhs_inv @ rhs_admm
        np.fill_diagonal(B, 0.0)
        # Z-update: soft-threshold + nonneg
        B_plus_U = B + U
        threshold = admm_l1 / admm_rho
        Z = np.maximum(B_plus_U - threshold, 0.0)
        np.fill_diagonal(Z, 0.0)
        # U-update
        U += B - Z
        # convergence
        resid = np.linalg.norm(B - Z) / (np.linalg.norm(B) + 1e-12)
        if resid < admm_tol:
            print(f"  converged at iter {it+1}, resid={resid:.2e}")
            break
    admm_W = Z
    admm_scores = x_dense @ admm_W
    np.savetxt(out_dir / "py_admmslim_W.csv", admm_W, delimiter=",")
    np.savetxt(out_dir / "py_admmslim_scores.csv", admm_scores, delimiter=",")
    admm_preds = topk_from_scores(admm_scores, train=train, k=10)
    admm_metrics = {
        "k": 10,
        "l1": admm_l1,
        "l2": admm_l2,
        "rho": admm_rho,
        "ndcg": ndcg_at_k(admm_preds, test, k=10),
        "recall": recall_at_k(admm_preds, test, k=10),
    }
    (out_dir / "py_admmslim_metrics.json").write_text(json.dumps(admm_metrics, indent=2), encoding="utf-8")
    print(f"  W nnz: {np.count_nonzero(admm_W)}, NDCG@10: {admm_metrics['ndcg']:.4f}")

    meta = {
        "seed": seed,
        "n_users": n_users,
        "n_items": n_items,
        "density": density,
        "rank": rank,
        "test_fraction": 0.2,
        "als_iters": als_iters,
        "ials_iters": 15,
        "eals_surrogate_iters": 10,
        "bpr_iters": bpr_iters,
        "bpr_class": bpr_cls.__name__,
        "lmf_available": lmf_scores is not None,
        "lmf_class": None if lmf_class is None else lmf_class.__name__,
        "ease_lambda": ease_lambda,
        "slim_available": slim_w is not None,
        "slim_l1": slim_l1,
        "slim_l2": slim_l2,
        "puresvd_rank": pure_rank,
        "knn_k": knn_k,
        "admm_l1": admm_l1,
        "admm_l2": admm_l2,
        "admm_rho": admm_rho,
        "implicit_version": getattr(implicit, "__version__", "unknown"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] Python fixtures generated:")
    print(f"  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
