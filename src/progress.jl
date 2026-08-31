# ──────────────────────────────────────────────────────────────────────────────
# Progress tracking and logging utilities
# ──────────────────────────────────────────────────────────────────────────────

"""
    ConvergenceMonitor

Tracks loss history and checks for convergence between iterations.
Also provides formatted progress logging.
"""
mutable struct ConvergenceMonitor{T<:AbstractFloat}
    losses::Vector{T}
    tol::T
    min_iter::Int
    start_ns::UInt64
end

function ConvergenceMonitor{T}(; tol::T=T(0.005), min_iter::Int=2) where {T}
    ConvergenceMonitor{T}(T[], tol, min_iter, time_ns())
end

"""
    record!(monitor, loss) -> Bool

Record a loss value and return `true` if converged.
"""
function record!(m::ConvergenceMonitor{T}, loss::T) where {T}
    push!(m.losses, loss)
    n = length(m.losses)
    n < m.min_iter && return false
    m.tol < zero(T) && return false  # negative tol disables convergence check
    prev = m.losses[n-1]
    rel_change = abs(prev - loss) / (abs(prev) + T(1e-12))
    return rel_change < m.tol
end

"""
    elapsed_seconds(monitor) -> Float64

Return elapsed wall-clock seconds since monitor creation.
"""
function elapsed_seconds(m::ConvergenceMonitor)
    (time_ns() - m.start_ns) / 1e9
end

"""
    iter_elapsed_str(monitor) -> String

Format elapsed time as a human-friendly string.
"""
function elapsed_str(seconds::Real)
    seconds = Float64(seconds)
    if seconds < 60.0
        @sprintf("%.2fs", seconds)
    elseif seconds < 3600.0
        @sprintf("%.1fmin", seconds / 60.0)
    else
        @sprintf("%.1fh", seconds / 3600.0)
    end
end

"""
    log_iteration(name, iter, max_iter, loss, iter_seconds, total_seconds; extra...)

Emit a structured @info log for a training iteration.
"""
function log_iteration(name::String, iter::Int, max_iter::Int,
                       loss::Real, iter_seconds::Real, total_seconds::Real;
                       extra::String="")
    pct = round(100.0 * iter / max_iter, digits=1)
    msg = @sprintf("[%s] iter %d/%d (%.1f%%) | loss=%.6f | iter=%s | total=%s",
                   name, iter, max_iter, pct, loss,
                   elapsed_str(iter_seconds), elapsed_str(total_seconds))
    if !isempty(extra)
        msg *= " | " * extra
    end
    @info msg
end
