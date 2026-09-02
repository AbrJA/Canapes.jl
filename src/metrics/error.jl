# ──────────────────────────────────────────────────────────────────────────────
# Error metrics for explicit feedback — RMSE and MAE
# ──────────────────────────────────────────────────────────────────────────────
#
# Convention (mirrors ranking.jl):
#   predictions : Matrix{T} of shape (n_users × n_items) — predicted ratings
#   actual      : SparseMatrixCSC (n_users × n_items) — observed ratings
#
# Errors are averaged over the OBSERVED entries of `actual` only (unobserved
# entries are not counted). These metrics are for the explicit subsystem
# (AbstractExplicitModel) and pair with `score`/`predict`.
# ──────────────────────────────────────────────────────────────────────────────

function _validate_error_inputs(predictions::AbstractMatrix{<:Real},
                                actual::SparseMatrixCSC)
    size(predictions) == size(actual) || throw(DimensionMismatch(
        "predictions has size $(size(predictions)) but actual has $(size(actual))"))
    nnz(actual) > 0 || throw(ArgumentError("actual contains no observed ratings to evaluate"))
    all(isfinite, predictions) || throw(ArgumentError(
        "predictions contain non-finite values"))
    nothing
end

function _observed_errors(predictions::AbstractMatrix{<:Real},
                          actual::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n = nnz(actual)
    errs = Vector{Float64}(undef, n)
    rv = rowvals(actual)
    nz = nonzeros(actual)
    pos = 1
    @inbounds for j in axes(actual, 2)
        for idx in nzrange(actual, j)
            u = rv[idx]
            errs[pos] = Float64(nz[idx]) - Float64(predictions[u, j])
            pos += 1
        end
    end
    errs
end

"""
    rmse(predictions::AbstractMatrix{<:Real}, actual::SparseMatrixCSC) -> Float64

Root Mean Squared Error over the observed entries of `actual`
(`predictions` is the dense n_users × n_items rating matrix).

# Example
```julia
julia> using SparseArrays

julia> actual = sparse([1,1,2,3], [1,2,1,4], [4.0,2.0,5.0,3.0], 3, 4);

julia> preds = fill(3.0, 3, 4);

julia> round(rmse(preds, actual); digits=4)
1.2247
```
"""
function rmse(predictions::AbstractMatrix{<:Real}, actual::SparseMatrixCSC)
    _validate_error_inputs(predictions, actual)
    errs = _observed_errors(predictions, actual)
    sqrt(sum(abs2, errs) / length(errs))
end

"""
    mae(predictions::AbstractMatrix{<:Real}, actual::SparseMatrixCSC) -> Float64

Mean Absolute Error over the observed entries of `actual`
(`predictions` is the dense n_users × n_items rating matrix).
"""
function mae(predictions::AbstractMatrix{<:Real}, actual::SparseMatrixCSC)
    _validate_error_inputs(predictions, actual)
    errs = _observed_errors(predictions, actual)
    sum(abs, errs) / length(errs)
end

# Macro-averaged aliases, mirroring the `mean_ap_at_k` naming convention.
# RMSE/MAE are already scalar aggregates; the aliases exist for API symmetry.
"""
    mean_rmse(predictions, actual) -> Float64

Alias for [`rmse`](@ref) (already macro-averaged). Kept for symmetry with the
`mean_*` family of the ranking metrics.
"""
mean_rmse(predictions::AbstractMatrix{<:Real}, actual::SparseMatrixCSC) =
    rmse(predictions, actual)

"""
    mean_mae(predictions, actual) -> Float64

Alias for [`mae`](@ref) (already macro-averaged). Kept for symmetry with the
`mean_*` family of the ranking metrics.
"""
mean_mae(predictions::AbstractMatrix{<:Real}, actual::SparseMatrixCSC) =
    mae(predictions, actual)