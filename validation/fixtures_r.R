#!/usr/bin/env Rscript
# validation/fixtures_r.R
# Generates R reference fixtures for Canapes.jl validation (NOT part of test suite).
# Run from project root: uvr run validation/fixtures_r.R
#                       (or: Rscript validation/fixtures_r.R)
#
# Outputs to /tmp/canapes_fixtures/ by default (directory is created if absent).
# You can override with env var CANAPES_R_FIXTURE_DIR.
# These CSV files are used by validation/validate_r.jl for reference comparison.
# See validation/README.md for details.

suppressPackageStartupMessages({
  library(rsparse)
  library(Matrix)
  library(MatrixExtra)
})

fixture_dir <- Sys.getenv("CANAPES_R_FIXTURE_DIR", "/tmp/canapes_fixtures")
dir.create(fixture_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(42)

RANK   <- 5L
LAMBDA <- 0.1
ALPHA  <- 1.0
N_ITER <- 50L   # enough iterations to reach convergence

cat("Generating Canapes.jl correctness fixtures...\n\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

write_triplet <- function(X, path) {
  cx <- as(X, "TsparseMatrix")
  write.csv(data.frame(row = cx@i + 1L, col = cx@j + 1L, val = cx@x),
            path, row.names = FALSE)
}

# Observed-entry implicit WRMF loss — mirrors Julia's _compute_loss exactly.
# U: n_users × rank (R convention)
# V: rank × n_items (R convention, = model$components)
obs_loss <- function(U, V, X, lambda, alpha) {
  cx  <- as(X, "TsparseMatrix")
  ui  <- cx@i + 1L
  ij  <- cx@j + 1L
  rv  <- cx@x
  # vectorised dot products: preds[k] = U[ui[k], ] · V[, ij[k]]
  preds <- rowSums(U[ui, , drop = FALSE] * t(V)[ij, , drop = FALSE])
  conf  <- 1.0 + alpha * rv
  sum(conf * (1.0 - preds)^2) + lambda * (sum(U^2) + sum(V^2))
}

has_model <- function(name) {
  exists(name, mode = "function") || exists(name, mode = "environment")
}

write_capabilities <- function(path) {
  model_names <- c(
    "WRMF", "FTRL", "GloVe", "FM", "FactorizationMachine",
    "PureSVD", "LinearFlow", "RankMF", "ScaleNormalize"
  )
  cap <- data.frame(
    model = model_names,
    available = as.integer(vapply(model_names, has_model, logical(1))),
    stringsAsFactors = FALSE
  )
  write.csv(cap, path, row.names = FALSE, quote = TRUE)
}

write_capabilities(file.path(fixture_dir, "rsparse_capabilities.csv"))

# ── 1. Shared small matrix: 100 users × 80 items, density=5%, ratings 1-5 ────
cat("1. Generating shared matrix (100x80, density=5%)...\n")
n_u <- 100L; n_i <- 80L
X <- rsparsematrix(n_u, n_i, density = 0.05,
                   rand.x = function(n) as.numeric(sample(1:5, n, TRUE)))
X <- as(X, "CsparseMatrix")
write_triplet(X, file.path(fixture_dir, "X_small.csv"))
write.csv(data.frame(nr = n_u, nc = n_i),
          file.path(fixture_dir, "X_small_dims.csv"), row.names = FALSE)
cat(sprintf("   nnz=%d\n", nnzero(X)))

# ── 2. WRMF Cholesky ──────────────────────────────────────────────────────────
cat("2. WRMF Cholesky (50 iter)...\n")
m_chol <- WRMF$new(rank = RANK, lambda = LAMBDA, alpha = ALPHA,
                   feedback = "implicit", solver = "cholesky")
U_chol <- m_chol$fit_transform(X, n_iter = N_ITER, convergence_tol = -1,
                                verbose = FALSE)
V_chol <- m_chol$components   # rank × n_items

write.csv(U_chol, file.path(fixture_dir, "wrmf_chol_user.csv"), row.names = FALSE)
write.csv(V_chol, file.path(fixture_dir, "wrmf_chol_item.csv"), row.names = FALSE)

r_loss_chol <- obs_loss(U_chol, V_chol, X, LAMBDA, ALPHA)
writeLines(as.character(r_loss_chol), file.path(fixture_dir, "wrmf_chol_loss.txt"))
cat(sprintf("   obs-entry loss: %.6f\n", r_loss_chol))

# Save score matrix for first 10 users (for prediction comparison)
scores_chol <- U_chol[1:10, ] %*% V_chol   # 10 × n_items
write.csv(scores_chol, file.path(fixture_dir, "wrmf_chol_scores_top10.csv"),
          row.names = FALSE)

# ── 3. WRMF Conjugate Gradient ────────────────────────────────────────────────
cat("3. WRMF CG (50 iter)...\n")
m_cg <- WRMF$new(rank = RANK, lambda = LAMBDA, alpha = ALPHA,
                 feedback = "implicit", solver = "conjugate_gradient")
U_cg <- m_cg$fit_transform(X, n_iter = N_ITER, convergence_tol = -1,
                             verbose = FALSE)
V_cg <- m_cg$components

write.csv(U_cg, file.path(fixture_dir, "wrmf_cg_user.csv"), row.names = FALSE)
write.csv(V_cg, file.path(fixture_dir, "wrmf_cg_item.csv"), row.names = FALSE)

r_loss_cg <- obs_loss(U_cg, V_cg, X, LAMBDA, ALPHA)
writeLines(as.character(r_loss_cg), file.path(fixture_dir, "wrmf_cg_loss.txt"))
cat(sprintf("   obs-entry loss: %.6f\n", r_loss_cg))

# ── 4. FTRL ───────────────────────────────────────────────────────────────────
cat("4. FTRL (5 epochs, 500x100)...\n")
set.seed(42)
n_f <- 500L; p_f <- 100L
X_ftrl <- rsparsematrix(n_f, p_f, density = 0.1)
X_ftrl <- as(X_ftrl, "RsparseMatrix")

w_true      <- numeric(p_f); w_true[1:5] <- 1
logit_fn    <- function(x) 1 / (1 + exp(-x))
y_ftrl      <- as.numeric(logit_fn(as.vector(X_ftrl %*% w_true)) > 0.5)

write_triplet(as(X_ftrl, "CsparseMatrix"), file.path(fixture_dir, "X_ftrl.csv"))
write.csv(data.frame(nr = n_f, nc = p_f),
          file.path(fixture_dir, "X_ftrl_dims.csv"), row.names = FALSE)
write.csv(data.frame(y = y_ftrl), file.path(fixture_dir, "y_ftrl.csv"), row.names = FALSE)

ftrl_m <- FTRL$new(learning_rate = 0.1, learning_rate_decay = 0.5,
                    lambda = 0.01, l1_ratio = 0.5, dropout = 0)
for (i in 1:5) ftrl_m$partial_fit(X_ftrl, y_ftrl)

w_r <- ftrl_m$coef()
p_r <- ftrl_m$predict(X_ftrl)
write.csv(data.frame(w = w_r), file.path(fixture_dir, "ftrl_weights.csv"), row.names = FALSE)
write.csv(data.frame(p = p_r), file.path(fixture_dir, "ftrl_preds.csv"),   row.names = FALSE)
cat(sprintf("   acc=%.4f  nnz_weights=%d/%d\n",
            mean(round(p_r) == y_ftrl), sum(w_r != 0), p_f))

# ── 5. Factorization Machine — XOR ──────────────────────────────────────────
cat("5. FM XOR (200 iter)...\n")
x_xor <- matrix(c(0,0, 0,1, 1,0, 1,1), nrow = 4, byrow = TRUE)
x_xor <- as(x_xor, "RsparseMatrix")
y_xor <- c(0.0, 1.0, 1.0, 0.0)

fm_m <- FactorizationMachine$new(
  learning_rate_w = 10, rank = 2, lambda_w = 0, lambda_v = 0,
  family = "binomial", intercept = TRUE, learning_rate_v = 10)
fm_m$fit(x_xor, y_xor, n_iter = 200)
p_fm <- fm_m$predict(x_xor)

write.csv(data.frame(p = p_fm), file.path(fixture_dir, "fm_xor_preds.csv"),
          row.names = FALSE)
cat("   model class: FactorizationMachine\n")
cat(sprintf("   preds: %.4f %.4f %.4f %.4f\n",
            p_fm[1], p_fm[2], p_fm[3], p_fm[4]))
cat(sprintf("   XOR correct: %s\n",
            all(p_fm[c(1,4)] < 0.3) && all(p_fm[c(2,3)] > 0.7)))

# ── 6. GloVe ──────────────────────────────────────────────────────────────────
cat("6. GloVe (30 iter, 50x50)...\n")
set.seed(42)
co <- rsparsematrix(50, 50, density = 0.1,
                    rand.x = function(n) runif(n, 0.1, 10))
co <- as(co + t(co), "CsparseMatrix")
co@x <- abs(co@x) + 0.1

write_triplet(co, file.path(fixture_dir, "glove_X.csv"))
write.csv(data.frame(nr = 50L, nc = 50L),
          file.path(fixture_dir, "glove_dims.csv"), row.names = FALSE)

glove_m <- GloVe$new(rank = 5, x_max = 10, learning_rate = 0.15)
glove_m$fit_transform(co, n_iter = 30, n_threads = 1, verbose = FALSE)
cost_hist <- glove_m$get_history()$cost_history
final_cost <- tail(cost_hist, 1)
writeLines(as.character(final_cost), file.path(fixture_dir, "glove_final_cost.txt"))
# Save full cost history for monotonicity comparison
write.csv(data.frame(cost = cost_hist), file.path(fixture_dir, "glove_costs.csv"),
          row.names = FALSE)
cat(sprintf("   final cost: %.6f  (over %d epochs)\n", final_cost, length(cost_hist)))

# ── 7. Metrics reference ──────────────────────────────────────────────────────
cat("7. Ranking metrics...\n")
actual_m <- sparseMatrix(i = c(1,1,1), j = c(5,7,9), x = c(1,1,1), dims = c(1,10))
preds_m  <- matrix(c(5L, 7L, 9L, 2L), nrow = 1)
ap_val   <- rsparse::ap_k(preds_m,   actual_m)[1]
ndcg_val <- rsparse::ndcg_k(preds_m, actual_m)[1]
write.csv(data.frame(ap = ap_val, ndcg = ndcg_val),
          file.path(fixture_dir, "metrics_ref.csv"), row.names = FALSE)
cat(sprintf("   AP@4=%.6f  NDCG@4=%.6f\n", ap_val, ndcg_val))

# ── 8. SoftImpute (soft_impute mode) ─────────────────────────────────────────
cat("8. SoftImpute (soft_impute target, 100 iter)...\n")
set.seed(42)
# Use the shared matrix X (100×80, density=5%, ratings 1-5).
# Run enough iterations to converge, then save results.
si_rank   <- 5L
si_lambda <- 1.0
si_niter  <- 100L

si_result <- soft_impute(X, rank = si_rank, lambda = si_lambda,
                         n_iter = si_niter, convergence_tol = 1e-6,
                         final_svd = TRUE)

# Singular values
write.csv(data.frame(d = si_result$d),
          file.path(fixture_dir, "softimpute_si_d.csv"), row.names = FALSE)

# Reconstruction at observed positions: U * diag(d) * V'
# si_result$u: m × rank, si_result$v: n × rank, si_result$d: rank-vector
si_recon <- si_result$u %*% diag(si_result$d) %*% t(si_result$v)
cx <- as(X, "TsparseMatrix")
si_obs_preds <- si_recon[cbind(cx@i + 1L, cx@j + 1L)]
write.csv(data.frame(pred = si_obs_preds),
          file.path(fixture_dir, "softimpute_si_obs_preds.csv"), row.names = FALSE)

# Frobenius norm of solution
si_frob <- sum(si_result$d^2)
writeLines(as.character(si_frob),
           file.path(fixture_dir, "softimpute_si_frob.txt"))
cat(sprintf("   rank=%d  frob=%.6f  d=[%s]\n",
            length(si_result$d), si_frob,
            paste(round(si_result$d, 4), collapse = ", ")))

# ── 9. SoftSVD (svd mode) ────────────────────────────────────────────────────
cat("9. SoftSVD (svd target, 100 iter)...\n")
set.seed(42)
svd_result <- soft_svd(X, rank = si_rank, lambda = si_lambda,
                       n_iter = si_niter, convergence_tol = 1e-6,
                       final_svd = TRUE)

write.csv(data.frame(d = svd_result$d),
          file.path(fixture_dir, "softimpute_svd_d.csv"), row.names = FALSE)

svd_recon <- svd_result$u %*% diag(svd_result$d) %*% t(svd_result$v)
svd_obs_preds <- svd_recon[cbind(cx@i + 1L, cx@j + 1L)]
write.csv(data.frame(pred = svd_obs_preds),
          file.path(fixture_dir, "softimpute_svd_obs_preds.csv"), row.names = FALSE)

svd_frob <- sum(svd_result$d^2)
writeLines(as.character(svd_frob),
           file.path(fixture_dir, "softimpute_svd_frob.txt"))
cat(sprintf("   rank=%d  frob=%.6f  d=[%s]\n",
            length(svd_result$d), svd_frob,
            paste(round(svd_result$d, 4), collapse = ", ")))

# ── 10. FM sparse high-dimensional (one-hot user/item, latent interaction) ───
cat("10. FM sparse high-dimensional (4000x160, 100 iter)...\n")
n_users <- 80L
n_items <- 80L
p_fm <- n_users + n_items
k_fm <- 2L
n_fm <- 4000L

# Well-posed regime: one-hot (user, item) rows whose target is a rank-k latent
# interaction y_ui = w0 + Σ_f V[f,u]·V[f,i] + noise. 62.5% pair coverage — a
# correct FM must recover the structure (oracle cor ≈ 1.0; SoftSVD reaches 0.95).
set.seed(321)
V_user <- matrix(rnorm(n_users * k_fm, 0, 1), nrow = k_fm)   # k × n_users
V_item <- matrix(rnorm(n_items * k_fm, 0, 1), nrow = k_fm)   # k × n_items
all_pairs <- cbind(rep(1:n_users, each = n_items), rep(1:n_items, n_users))
set.seed(322)
sel <- sample.int(nrow(all_pairs), n_fm, replace = FALSE)    # unique pairs
u_idx <- all_pairs[sel, 1]
i_idx <- all_pairs[sel, 2]
X_fm <- sparseMatrix(i = rep(1:n_fm, each = 2),
                     j = as.vector(rbind(u_idx, n_users + i_idx)),
                     x = 1, dims = c(n_fm, p_fm))
y_fm <- 0.1 + colSums(V_user[, u_idx] * V_item[, i_idx]) + rnorm(n_fm, 0, 0.05)
write_triplet(X_fm, file.path(fixture_dir, "fm_sparse_X.csv"))
write.csv(data.frame(nr = n_fm, nc = p_fm),
          file.path(fixture_dir, "fm_sparse_dims.csv"), row.names = FALSE)
write.csv(data.frame(y = y_fm), file.path(fixture_dir, "fm_sparse_y.csv"),
          row.names = FALSE)
# Save the ground-truth latent factors so Julia can compute the oracle bound.
write.csv(t(V_user), file.path(fixture_dir, "fm_sparse_V_user.csv"), row.names = FALSE)
write.csv(t(V_item), file.path(fixture_dir, "fm_sparse_V_item.csv"), row.names = FALSE)

# Seed the RNG right before fitting: rsparse's FM init draws unseeded runif
# (fill_float_vector_randn), which made the fixture nondeterministic across runs.
set.seed(4242)
fm_r <- FactorizationMachine$new(learning_rate_w = 0.2, rank = k_fm,
                                 lambda_w = 0, lambda_v = 0,
                                 family = "gaussian", intercept = TRUE)
fm_r$fit(X_fm, y_fm, n_iter = 100)
preds_fm <- fm_r$predict(X_fm)
write.csv(data.frame(p = preds_fm), file.path(fixture_dir, "fm_sparse_preds.csv"),
          row.names = FALSE)
cat(sprintf("   cor(preds, y)=%.4f\n", cor(preds_fm, y_fm)))

# ── Summary ───────────────────────────────────────────────────────────────────
cat(sprintf("\nFixtures written to %s:\n", fixture_dir))
for (f in sort(list.files(fixture_dir))) {
  sz <- file.info(file.path(fixture_dir, f))$size
  cat(sprintf("  %-40s  %6.1f KB\n", f, sz / 1024))
}
cat("\nRun Canapes.jl tests with: julia --project=. --threads=4,2 -e 'using Pkg; Pkg.test()'\n")
