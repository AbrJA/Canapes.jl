# validation/common.jl
# Shared helpers for the R / Python reference validation scripts.
# Included by validate_r.jl and validate_py.jl (see run.jl for the runner).

using Test, SparseArrays, LinearAlgebra, Random, Statistics

# ── Fixture directories (overridable via env, same defaults as before) ────────
const R_FIXTURE_DIR  = get(ENV, "GIDEON_R_FIXTURE_DIR", "/tmp/gideon_fixtures")
const PY_FIXTURE_DIR = get(ENV, "GIDEON_PY_FIXTURE_DIR", "/tmp/gideon_fixtures/python")

# ── Threshold helpers ─────────────────────────────────────────────────────────

"""Parse a Float64 from an env var, falling back to `default`."""
function _env_float(name::String, default::Float64)
    parse(Float64, get(ENV, name, string(default)))
end

"""True if every path in `paths` exists."""
_all_files_exist(paths::Vector{String}) = all(isfile, paths)

"""Require fixture files; print a clear error and exit non-zero if missing."""
function require_files(paths::Vector{String}, dir::String, hint::String)
    missing = [p for p in paths if !isfile(p)]
    isempty(missing) && return true
    println(stderr, """
    ERROR: missing required fixture files in $dir:
      - $(join(missing, "\n  - "))
    $hint
    """)
    exit(1)
end

# ── CSV / triplet readers (formats produced by fixtures_r.R / fixtures_py.py) ─

"""Parse a single Float64 scalar from a plain-text file."""
function _read_scalar(path::String)
    parse(Float64, strip(read(path, String)))
end

"""Read a named column from a CSV file produced by R's write.csv."""
function _read_col(path::String, col::String)
    lines = readlines(path)
    header = [strip(h, '"') for h in split(lines[1], ',')]
    idx    = findfirst(==(col), header)
    isnothing(idx) &&
        error("Column '$col' not in $(basename(path)). Found: $(join(header, ", "))")
    [parse(Float64, strip(split(lines[k], ',')[idx], '"'))
     for k in 2:length(lines) if !isempty(strip(lines[k]))]
end

"""Read a full CSV matrix (rows × cols). Skips a header row when present
   (R's write.csv writes one; np.savetxt does not)."""
function _read_matrix(path::String)
    lines = readlines(path)
    if !isempty(lines)
        first_row = split(strip(lines[1]), ',')
        if any(f -> isempty(f) ||
                   isnothing(tryparse(Float64, strip(f, '"'))), first_row)
            lines = lines[2:end]   # header row detected
        end
    end
    rows  = [parse.(Float64, split(strip(l), ','))
             for l in lines if !isempty(strip(l))]
    isempty(rows) && return Matrix{Float64}(undef, 0, 0)
    reduce(vcat, [r' for r in rows])
end

"""Load a sparse matrix from a (row,col,val) triplet CSV + dims CSV."""
function _load_sparse(triplet::String, dims::String)
    lines = readlines(triplet)
    rv = Int[]; cv = Int[]; vv = Float64[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        p = split(strip(line), ',')
        push!(rv, parse(Int, p[1])); push!(cv, parse(Int, p[2])); push!(vv, parse(Float64, p[3]))
    end
    dlines = readlines(dims)
    hdr    = [strip(h, '"') for h in split(dlines[1], ',')]
    dvals  = split(strip(dlines[2]), ',')
    nr     = parse(Int, dvals[findfirst(==("nr"), hdr)])
    nc     = parse(Int, dvals[findfirst(==("nc"), hdr)])
    sparse(rv, cv, vv, nr, nc)
end

# ── Statistics ────────────────────────────────────────────────────────────────

"""Pearson correlation (stdlib only)."""
function _cor(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    x64 = Float64.(x); y64 = Float64.(y)
    mx = sum(x64) / length(x64); my = sum(y64) / length(y64)
    dx = x64 .- mx; dy = y64 .- my
    dot(dx, dy) / (sqrt(dot(dx, dx) * dot(dy, dy)) + 1e-15)
end

"""Top-k item indices per user (1-based), highest score first."""
function _row_topk(scores::AbstractMatrix{<:Real}, k::Int)
    n_users = size(scores, 1)
    out = Matrix{Int}(undef, n_users, k)
    for u in 1:n_users
        out[u, :] = partialsortperm(@view(scores[u, :]), 1:k; rev=true)
    end
    out
end

"""Mean Jaccard overlap of per-user top-k lists."""
function _mean_topk_overlap(a::Matrix{Int}, b::Matrix{Int})
    n_users = size(a, 1)
    k = size(a, 2)
    s = 0.0
    for u in 1:n_users
        s += length(intersect(@view(a[u, :]), @view(b[u, :]))) / k
    end
    s / n_users
end

"""`recommend` returning an (n_users, k) matrix regardless of orientation."""
function _safe_recommend(model, X::SparseMatrixCSC, k::Int)
    n_users, n_items = size(X)
    r = recommend(model, X; k=k)
    size(r) == (n_users, k) && return r
    size(r, 1) == k && size(r, 2) == n_users && return Matrix(r')
    error("Unexpected recommend() shape $(size(r)) for n_users=$n_users, k=$k")
end

"""Read a numeric value from the flat JSON metric files written by fixtures_py.py."""
function _read_json_metric(path::String, key::String)
    txt = read(path, String)
    m = match(Regex("\"$key\"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)"), txt)
    isnothing(m) && error("Metric '$key' not found in $(basename(path))")
    parse(Float64, m.captures[1])
end

# ── Parity helpers ────────────────────────────────────────────────────────────

"""
    assert_metric_parity(; label, model, X_train, X_test, metrics_path,
                         max_ndcg, max_recall, k=10)

Compare ranking metrics (NDCG@k, Recall@k) computed from `recommend(model, X_train)`
against Python reference values stored in `metrics_path`. Enforced via `@test` when
a threshold is given; diagnostic (`@info`) when the threshold is `nothing`.
"""
function assert_metric_parity(; label, model, X_train, X_test, metrics_path,
                              max_ndcg=nothing, max_recall=nothing, k::Int=10)
    if !isfile(metrics_path)
        @info "Skipping $label split-metric parity: $(basename(metrics_path)) not found"
        return
    end
    py_ndcg = _read_json_metric(metrics_path, "ndcg")
    py_recall = _read_json_metric(metrics_path, "recall")
    jl_preds = _safe_recommend(model, X_train, k)
    jl_ndcg = mean(ndcg_at_k(jl_preds, X_test; k=k))
    jl_recall = mean(recall_at_k(jl_preds, X_test; k=k))
    println("  $label NDCG@$k: Julia=$jl_ndcg, Python=$py_ndcg, Δ=$(abs(jl_ndcg - py_ndcg))")
    println("  $label Recall@$k: Julia=$jl_recall, Python=$py_recall, Δ=$(abs(jl_recall - py_recall))")
    if isnothing(max_ndcg) && isnothing(max_recall)
        @info "$label split-metric parity is diagnostic; set thresholds to enforce"
    else
        !isnothing(max_ndcg) && @test abs(jl_ndcg - py_ndcg) <= max_ndcg
        !isnothing(max_recall) && @test abs(jl_recall - py_recall) <= max_recall
    end
end

"""
    assert_score_parity(; label, model, X, py_scores, min_cor, min_overlap,
                        k=10, n_eval=25, X_train=nothing, X_test=nothing,
                        metrics_path=nothing, max_ndcg=nothing, max_recall=nothing)

Fit `model` on `X`, compare the full score matrix against a Python reference
(pearson correlation + mean top-k overlap), and optionally compare ranking
metrics on a train/test split. All comparisons are `@test`ed.
"""
function assert_score_parity(; label, model, X, py_scores,
                             min_cor, min_overlap,
                             k::Int=10, n_eval::Int=25,
                             X_train=nothing, X_test=nothing,
                             metrics_path=nothing,
                             max_ndcg=nothing, max_recall=nothing)
    fit!(model, X; rng=MersenneTwister(42))
    jl_scores = Matrix(transpose(model.user_factors) * model.item_factors)

    c = _cor(vec(jl_scores), vec(py_scores))
    jl_top = _row_topk(jl_scores[1:n_eval, :], k)
    py_top = _row_topk(py_scores[1:n_eval, :], k)
    overlap = _mean_topk_overlap(jl_top, py_top)

    @test isfinite(c)
    @test c >= min_cor
    @test overlap >= min_overlap
    println("  $label score correlation: $c")
    println("  $label top-$k overlap: $overlap")

    if !isnothing(metrics_path)
        assert_metric_parity(; label, model, X_train, X_test, metrics_path,
                             max_ndcg, max_recall, k)
    end
    return (cor=c, overlap=overlap)
end
