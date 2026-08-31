# ──────────────────────────────────────────────────────────────────────────────
# Tables.jl integration — accept interaction data as (user, item, value) triplets
# ──────────────────────────────────────────────────────────────────────────────

"""
    interactions_to_sparse(table; user_col=:user, item_col=:item, value_col=:value,
                           n_users=nothing, n_items=nothing, T=Float64) -> SparseMatrixCSC

Convert any Tables.jl-compatible source (DataFrames, CSV, Arrow, NamedTuples of
vectors, vectors of named tuples, …) to a sparse user-item interaction matrix.

Columns `user_col` and `item_col` must contain 1-based integer indices. The
`value_col` column provides the interaction strength (1.0 when `nothing` or
missing). Repeated (user, item) pairs are accumulated by summing their values,
matching `sparse` semantics.

# Arguments
- `table` — any Tables.jl-compatible table
- `user_col::Symbol` — name of the user ID column
- `item_col::Symbol` — name of the item ID column
- `value_col::Symbol` — name of the value/rating column (use `nothing` for implicit=1)
- `n_users::Union{Nothing,Int}` — number of users (auto-detected if nothing)
- `n_items::Union{Nothing,Int}` — number of items (auto-detected if nothing)
- `T::Type{<:AbstractFloat}` — element type of the returned matrix

# Example
```julia
julia> using Tables

julia> df = (user=[1,1,2,3,3,3], item=[2,5,3,1,2,4], rating=[1.0,1.0,1.0,1.0,1.0,1.0]);

julia> X = interactions_to_sparse(df; user_col=:user, item_col=:item, value_col=:rating);

julia> size(X)
(3, 5)
```
"""
function interactions_to_sparse(table;
                                user_col::Symbol = :user,
                                item_col::Symbol = :item,
                                value_col::Union{Symbol,Nothing} = :value,
                                n_users::Union{Nothing,Int} = nothing,
                                n_items::Union{Nothing,Int} = nothing,
                                T::Type{<:AbstractFloat} = Float64)
    Tables.istable(table) || throw(ArgumentError(
        "expected a Tables.jl-compatible table, got $(typeof(table))"))

    ct = Tables.columns(table)
    cols = Tables.columnnames(ct)
    user_col in cols || throw(ArgumentError(
        "table has no column $user_col; available columns: $(collect(cols))"))
    item_col in cols || throw(ArgumentError(
        "table has no column $item_col; available columns: $(collect(cols))"))

    users = _as_indices(Tables.getcolumn(ct, user_col), user_col)
    items = _as_indices(Tables.getcolumn(ct, item_col), item_col)
    length(users) == length(items) || throw(DimensionMismatch(
        "columns $user_col and $item_col have different lengths " *
        "($(length(users)) vs $(length(items)))"))

    isempty(users) && throw(ArgumentError("interaction table is empty"))

    vals = if value_col === nothing
        ones(T, length(users))
    else
        value_col in cols || throw(ArgumentError(
            "table has no column $value_col; available columns: $(collect(cols))"))
        _as_values(Tables.getcolumn(ct, value_col), value_col, T)
    end

    nu = something(n_users, maximum(users))
    ni = something(n_items, maximum(items))

    _all_in_range(users, 1, nu) || throw(ArgumentError(
        "user indices must be in [1, $nu], got $(minimum(users))..$(maximum(users))"))
    _all_in_range(items, 1, ni) || throw(ArgumentError(
        "item indices must be in [1, $ni], got $(minimum(items))..$(maximum(items))"))

    # Repeated (user, item) pairs accumulate by summing their values
    sparse(users, items, vals, nu, ni)
end

"""
    sparse_to_interactions(X::SparseMatrixCSC) -> NamedTuple

Convert a sparse matrix back to (user, item, value) triplet vectors.
Returns a NamedTuple with fields `:user`, `:item`, `:value`.

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 10, 5, 0.3);

julia> triplets = sparse_to_interactions(X);

julia> length(triplets.user) == nnz(X)
true
```
"""
function sparse_to_interactions(X::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    users = Int[]
    items = Int[]
    vals = Tv[]
    rv = rowvals(X)
    nz = nonzeros(X)
    for j in axes(X, 2)
        for idx in nzrange(X, j)
            push!(users, Int(rv[idx]))
            push!(items, j)
            push!(vals, nz[idx])
        end
    end
    (user=users, item=items, value=vals)
end

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Convert an ID column to Int indices, with clean errors for non-integer input.
function _as_indices(col, name::Symbol)
    if eltype(col) <: Integer
        return Int.(col)
    end
    try
        return Int.(col)
    catch
        throw(ArgumentError(
            "column $name must contain integer indices, got $(eltype(col)) values"))
    end
end

# Convert a value column to the requested float T.
function _as_values(col, name::Symbol, ::Type{T}) where {T<:AbstractFloat}
    if eltype(col) <: AbstractFloat
        return T.(col)
    elseif eltype(col) <: Integer
        return T.(col)
    end
    try
        return T.(col)
    catch
        throw(ArgumentError(
            "column $name is not convertible to $T values"))
    end
end

@inline function _all_in_range(v::Vector{Int}, lo::Int, hi::Int)
    @inbounds for x in v
        (lo <= x <= hi) || return false
    end
    true
end
