# ──────────────────────────────────────────────────────────────────────────────
# Ranking evaluation metrics — MAP@K, NDCG@K, Precision@K, Recall@K
# ──────────────────────────────────────────────────────────────────────────────
#
# All metrics follow the convention:
#   predictions : Matrix{Int} of shape (n_users, K)  — predicted item indices
#   actual      : SparseMatrixCSC (n_users × n_items) — non-zero = relevant
# ──────────────────────────────────────────────────────────────────────────────

function _validate_ranking_inputs(predictions::AbstractMatrix{<:Integer},
                                  actual::SparseMatrixCSC, k::Int)
    k >= 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    size(predictions, 2) >= 1 || throw(ArgumentError("predictions must contain at least one item per user"))
    n_users, n_items = size(actual)
    size(predictions, 1) == n_users || throw(DimensionMismatch(
        "predictions has $(size(predictions, 1)) users but actual has $n_users"))
    all(item -> 1 <= item <= n_items, predictions) || throw(ArgumentError(
        "predictions contain an item index outside 1:$n_items"))

    k_eff = min(k, size(predictions, 2))
    for u in axes(predictions, 1)
        length(Set(@view predictions[u, 1:k_eff])) == k_eff || throw(ArgumentError(
            "predictions contain duplicate items for user $u in the evaluated top-k"))
    end
    k_eff
end

# ──────────────── Average Precision @ K ────────────────

"""
    ap_at_k(predictions::AbstractMatrix{Int}, actual::SparseMatrixCSC; k=size(predictions,2))

Compute Average Precision @ K for each user.
Returns a vector of length `n_users`.
"""
function ap_at_k(predictions::AbstractMatrix{<:Integer},
                 actual::SparseMatrixCSC;
                  k::Int = size(predictions, 2))
    k_eff = _validate_ranking_inputs(predictions, actual, k)
    n_users = size(predictions, 1)
    actual_t = _transpose_for_row_access(actual)
    result = Vector{Float64}(undef, n_users)
    nt = Threads.nthreads()
    Threads.@threads for chunk in 1:nt
        for u in _thread_chunk_bounds(chunk, n_users, nt)
            @inbounds result[u] = _ap_single(predictions, actual_t, u, k_eff)
        end
    end
    result
end

"""
    mean_ap_at_k(predictions, actual; k)

Mean Average Precision @ K (macro-averaged over all users).
"""
function mean_ap_at_k(predictions::AbstractMatrix{<:Integer},
                  actual::SparseMatrixCSC;
                  k::Int = size(predictions, 2))
    aps = ap_at_k(predictions, actual; k)
    isempty(aps) ? 0.0 : sum(aps) / length(aps)
end

function _ap_single(predictions::AbstractMatrix{<:Integer},
                    actual::SparseMatrixCSC, u::Int, k::Int)
    relevant = Set(_relevant_items(actual, u))
    isempty(relevant) && return 0.0

    k_eff = min(k, size(predictions, 2))
    hits = 0
    cum_precision = 0.0
    @inbounds for pos in 1:k_eff
        item = predictions[u, pos]
        if item in relevant
            hits += 1
            cum_precision += hits / pos
        end
    end
    cum_precision / min(k_eff, length(relevant))
end

# ──────────────── NDCG @ K ────────────────

"""
    ndcg_at_k(predictions::AbstractMatrix{Int}, actual::SparseMatrixCSC; k)

Normalized Discounted Cumulative Gain @ K for each user.
Non-zero values in `actual` are treated as relevance scores.
"""
function ndcg_at_k(predictions::AbstractMatrix{<:Integer},
                   actual::SparseMatrixCSC;
                   k::Int = size(predictions, 2))
    k_eff = _validate_ranking_inputs(predictions, actual, k)
    n_users = size(predictions, 1)
    actual_t = _transpose_for_row_access(actual)
    result = Vector{Float64}(undef, n_users)
    nt = Threads.nthreads()
    Threads.@threads for chunk in 1:nt
        for u in _thread_chunk_bounds(chunk, n_users, nt)
            @inbounds result[u] = _ndcg_single(predictions, actual_t, u, k_eff)
        end
    end
    result
end

function _ndcg_single(predictions::AbstractMatrix{<:Integer},
                      actual::SparseMatrixCSC, u::Int, k::Int)
    items, rels = _relevant_items_with_scores(actual, u)
    isempty(items) && return 0.0

    item_rel = Dict(zip(items, rels))
    k_eff = min(k, size(predictions, 2))

    # DCG
    dcg = 0.0
    @inbounds for pos in 1:k_eff
        item = predictions[u, pos]
        rel = get(item_rel, item, 0.0)
        dcg += rel / log2(pos + 1)
    end

    # IDCG
    sorted_rels = sort(rels; rev=true)
    idcg = 0.0
    kk = min(k_eff, length(sorted_rels))
    @inbounds for pos in 1:kk
        idcg += sorted_rels[pos] / log2(pos + 1)
    end

    idcg ≈ 0.0 ? 0.0 : dcg / idcg
end

# ──────────────── Precision @ K ────────────────

"""
    precision_at_k(predictions::AbstractMatrix{Int}, actual::SparseMatrixCSC; k)

Precision @ K for each user.
"""
function precision_at_k(predictions::AbstractMatrix{<:Integer},
                        actual::SparseMatrixCSC;
                        k::Int = size(predictions, 2))
    k_eff = _validate_ranking_inputs(predictions, actual, k)
    n_users = size(predictions, 1)
    actual_t = _transpose_for_row_access(actual)
    result = Vector{Float64}(undef, n_users)
    nt = Threads.nthreads()
    Threads.@threads for chunk in 1:nt
        for u in _thread_chunk_bounds(chunk, n_users, nt)
            @inbounds result[u] = _precision_single(predictions, actual_t, u, k_eff)
        end
    end
    result
end

function _precision_single(predictions::AbstractMatrix{<:Integer},
                           actual::SparseMatrixCSC, u::Int, k::Int)
    relevant = Set(_relevant_items(actual, u))
    isempty(relevant) && return 0.0

    k_eff = min(k, size(predictions, 2))
    hits = 0
    @inbounds for pos in 1:k_eff
        if predictions[u, pos] in relevant
            hits += 1
        end
    end
    hits / k_eff
end

# ──────────────── Recall @ K ────────────────

"""
    recall_at_k(predictions::AbstractMatrix{Int}, actual::SparseMatrixCSC; k)

Recall @ K for each user.
"""
function recall_at_k(predictions::AbstractMatrix{<:Integer},
                     actual::SparseMatrixCSC;
                     k::Int = size(predictions, 2))
    k_eff = _validate_ranking_inputs(predictions, actual, k)
    n_users = size(predictions, 1)
    actual_t = _transpose_for_row_access(actual)
    result = Vector{Float64}(undef, n_users)
    nt = Threads.nthreads()
    Threads.@threads for chunk in 1:nt
        for u in _thread_chunk_bounds(chunk, n_users, nt)
            @inbounds begin
                relevant = Set(_relevant_items(actual_t, u))
                if isempty(relevant)
                    result[u] = 0.0
                    continue
                end
                hits = 0
                for pos in 1:k_eff
                    if predictions[u, pos] in relevant
                        hits += 1
                    end
                end
                result[u] = hits / length(relevant)
            end
        end
    end
    result
end

# ──────────────── Helpers: extract relevant items from sparse row ────────────────

# Cache-friendly row extraction: transpose to CSC(Aᵀ) so row u = column u in Aᵀ.
# This is O(nnz_u) per user instead of O(nnz) for the CSC column scan.

function _transpose_for_row_access(actual::SparseMatrixCSC)
    # Aᵀ stored as CSC: column u of Aᵀ = row u of A
    SparseMatrixCSC(actual')
end

function _relevant_items(actual_t::SparseMatrixCSC, u::Int)
    # actual_t is the transpose: columns = original rows
    rv = rowvals(actual_t)
    collect(Int, @view rv[nzrange(actual_t, u)])
end

function _relevant_items_with_scores(actual_t::SparseMatrixCSC, u::Int)
    rv = rowvals(actual_t)
    nz = nonzeros(actual_t)
    rng = nzrange(actual_t, u)
    items = collect(Int, @view rv[rng])
    scores = [Float64(nz[idx]) for idx in rng]
    (items, scores)
end
