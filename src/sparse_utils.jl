# ──────────────────────────────────────────────────────────────────────────────
# Sparse matrix utilities — CSC / CSR helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    to_csr(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

Convert a CSC sparse matrix to CSR (SparseMatrixCSR) via transpose.
"""
function to_csr(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    At = SparseMatrixCSC(A')  # transpose → CSC of Aᵀ
    m, n = size(A)
    # CSR of A ≡ CSC of Aᵀ re-interpreted
    SparseMatrixCSR{1}(m, n, At.colptr, At.rowval, At.nzval)
end

"""
    sparse_row_norms(A::SparseMatrixCSC, p::Int=2)

Compute row-wise Lp norms of a CSC sparse matrix without allocating full dense rows.
"""
function sparse_row_norms(A::SparseMatrixCSC{Tv}, p::Int=2) where {Tv}
    p > 0 || throw(ArgumentError("p must be positive, got $p"))
    m = size(A, 1)
    norms = zeros(Tv, m)
    rv = rowvals(A)
    nz = nonzeros(A)
    @inbounds for col in axes(A, 2)
        for idx in nzrange(A, col)
            row = rv[idx]
            if p == 2
                norms[row] += nz[idx]^2
            elseif p == 1
                norms[row] += abs(nz[idx])
            else
                norms[row] += abs(nz[idx])^p
            end
        end
    end
    if p == 2
        @inbounds @simd for i in eachindex(norms)
            norms[i] = sqrt(norms[i])
        end
    elseif p != 1
        inv_p = one(Tv) / p
        @inbounds @simd for i in eachindex(norms)
            norms[i] = norms[i]^inv_p
        end
    end
    norms
end

"""
    sparse_col_nnz(A::SparseMatrixCSC)

Return a vector with the number of structural non-zeros per column.
"""
function sparse_col_nnz(A::SparseMatrixCSC)
    n = size(A, 2)
    counts = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        counts[j] = A.colptr[j+1] - A.colptr[j]
    end
    counts
end

"""
    sparse_row_nnz(A::SparseMatrixCSC)

Return a vector with the number of structural non-zeros per row.
"""
function sparse_row_nnz(A::SparseMatrixCSC)
    m = size(A, 1)
    counts = zeros(Int, m)
    rv = rowvals(A)
    @inbounds for col in axes(A, 2)
        for idx in nzrange(A, col)
            counts[rv[idx]] += 1
        end
    end
    counts
end

"""
    dual_representation(A::SparseMatrixCSC)

Return `(A, Aᵀ)` where `Aᵀ` is a fresh CSC copy of the transpose.
Useful for ALS algorithms that need fast row and column slicing.
"""
function dual_representation(A::SparseMatrixCSC)
    At = SparseMatrixCSC(A')
    (A, At)
end
