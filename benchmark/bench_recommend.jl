# benchmark/bench_recommend.jl
# Benchmark recommend / score / predict throughput and memory across all algorithms.
#
# Run:
#   julia --project -t4 benchmark/bench_recommend.jl
#
# The script trains each model once, then benchmarks the inference phase:
#   - recommend(model, X; k=10)   — top-k recommendation
#   - recommend(model, X; k=50)   — top-k with larger k
#   - score(model, X)             — full dense scoring
#   - predict(model, X)           — regression output (FTRL, FM)
#   - similar_items(model, id)    — item-to-item similarity
#
# Memory is tracked via RSS + peak RSS from /proc (Linux).
# Uses Float32 throughout to keep memory footprint manageable.

using Gideon, SparseArrays, Random, LinearAlgebra, Printf

# ─────────────────────────────────────────────
# Memory monitoring (Linux /proc-based)
# ─────────────────────────────────────────────

"""Read current process RSS in bytes from /proc/self/statm."""
function _rss_bytes()
    pagesize = ccall(:getpagesize, Cint, ())
    fields = split(read("/proc/self/statm", String))
    return parse(Int, fields[2]) * pagesize
end

"""Read peak RSS (VmHWM) from /proc/self/status."""
function _peak_rss_bytes()
    for line in eachline("/proc/self/status")
        if startswith(line, "VmHWM:")
            return parse(Int, split(line)[2]) * 1024  # kB → bytes
        end
    end
    return 0
end

function _fmt_bytes(b)
    b < 0          && return "0 B"
    b < 1024       && return @sprintf("%d B", b)
    b < 1024^2     && return @sprintf("%.1f KiB", b / 1024)
    b < 1024^3     && return @sprintf("%.1f MiB", b / 1024^2)
    return @sprintf("%.1f GiB", b / 1024^3)
end

"""Run `f()` three times, return (median_seconds, peak_rss_delta_bytes)."""
function bench(f; warmup::Bool=true, repeats::Int=3)
    warmup && f()  # JIT warmup
    GC.gc()

    times = Float64[]
    mem_deltas = Int[]
    for _ in 1:repeats
        GC.gc()
        rss_before = _rss_bytes()
        t = @elapsed result = f()
        rss_after = _rss_bytes()
        push!(times, t)
        push!(mem_deltas, max(0, rss_after - rss_before))
        result = nothing
        GC.gc()
    end
    median_t = sort(times)[cld(length(times), 2)]
    median_m = sort(mem_deltas)[cld(length(mem_deltas), 2)]
    return (time=median_t, mem=median_m)
end

# ─────────────────────────────────────────────
# Synthetic data generators (Float32)
# ─────────────────────────────────────────────

function generate_matrix(n_users::Int, n_items::Int, density::Float64; seed::Int=42)
    rng = MersenneTwister(seed)
    nnz_target = round(Int, n_users * n_items * density)
    rows = rand(rng, 1:n_users, nnz_target)
    cols = rand(rng, 1:n_items, nnz_target)
    vals = ones(Float32, nnz_target)
    sparse(rows, cols, vals, n_users, n_items)
end

"""Generate a regression-style sparse feature matrix + binary labels."""
function generate_regression_data(n_samples::Int, n_features::Int, density::Float64;
                                  seed::Int=42)
    rng = MersenneTwister(seed)
    nnz_target = round(Int, n_samples * n_features * density)
    rows = rand(rng, 1:n_samples, nnz_target)
    cols = rand(rng, 1:n_features, nnz_target)
    vals = Float32.(randn(rng, nnz_target))
    X = sparse(rows, cols, vals, n_samples, n_features)
    y = Float32.(rand(rng, Bool, n_samples))
    return X, y
end

# ─────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────

struct InferenceResult
    scale::String
    algorithm::String
    operation::String
    time_seconds::Float64
    memory_bytes::Int
    n_users::Int
    n_items::Int
    throughput::Float64   # users/sec or items/sec
end

# ─────────────────────────────────────────────
# Benchmark configurations
# ─────────────────────────────────────────────
#
# Scales chosen to avoid OOM on 16GB systems:
#   - score() output = n_users × n_items × 4 bytes (Float32)
#   - small:  1k × 500     → score =   2 MB
#   - medium: 10k × 2000   → score =  80 MB
#   - large:  50k × 5000   → score =   1 GB
#
# Item-similarity models store W (n_items × n_items):
#   - medium: 2000² × 4 = 16 MB
#   - large:  5000² × 4 = 100 MB  ← still fine

const SCALES = [
    (name="small",   n_users=1_000,   n_items=500,    density=0.05),
    (name="medium",  n_users=10_000,  n_items=2_000,  density=0.02),
    (name="large",   n_users=50_000,  n_items=5_000,  density=0.005),
]

"""Check if algorithm should be skipped at this scale."""
function skip_at_scale(algo::String, n_items::Int)
    # SLIM coordinate descent is O(n_items² × n_iter) — very slow above 2k items
    if algo == "SLIM" && n_items > 2_000
        return true
    end
    return false
end

"""Estimate score output size in bytes."""
_score_bytes(n_users, n_items) = Int64(n_users) * n_items * 4  # Float32

# ─────────────────────────────────────────────
# Model definitions (data is Float32, params are Float64)
# ─────────────────────────────────────────────

function mf_models()
    [
        ("WRMF-Chol",   () -> WMF(rank=64, λ=0.1, α=40.0, max_iter=5,
                                   solver=CholeskySolver(), verbose=false)),
        ("WRMF-CG",     () -> WMF(rank=64, λ=0.1, α=40.0, max_iter=5,
                                   solver=ConjugateGradient(), cg_steps=3, verbose=false)),
        ("iALS",        () -> IALS(rank=64, λ=0.1, α=40.0, max_iter=5, verbose=false)),
        ("eALS",        () -> EALS(rank=64, λ=10.0, w0=1.0, max_iter=5, verbose=false)),
        ("BPR",         () -> BPR(rank=64, λ_user=0.01, λ_pos=0.01, λ_neg=0.01,
                                  learning_rate=0.05, max_iter=5, verbose=false)),
        ("LMF",         () -> LogisticMF(rank=64, λ=0.6, learning_rate=1.0,
                                         max_iter=5, n_negative=10,
                                         convergence_tol=-1.0, verbose=false)),
        ("SoftImpute",  () -> SoftImpute(rank=64, λ=10.0, max_iter=5, verbose=false)),
        ("PureSVD",     () -> PureSVD(rank=64, max_iter=5, verbose=false)),
    ]
end

function similarity_models()
    [
        ("EASE",     () -> EASE(λ=500.0, verbose=false)),
        ("SLIM",     () -> SLIM(λ_1=0.01, λ_2=1.0, max_iter=30, verbose=false)),
        ("ADMMSLIM", () -> ADMMSLIM(λ_1=0.01, λ_2=100.0, max_iter=30, verbose=false)),
        ("ItemKNN",  () -> ItemKNN(k=50, similarity=:cosine, verbose=false)),
    ]
end

function regression_models()
    [
        ("FTRL",  () -> FTRL(learning_rate=0.1, λ=0.1, l1_ratio=0.5,
                              max_iter=5, verbose=false)),
        ("FM",    () -> FM(rank=8, λ_w=0.1, λ_v=0.1, learning_rate_w=0.01,
                           max_iter=5, verbose=false)),
    ]
end

# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────

function run_recommender_benchmarks(model, algo_name::String, X::SparseMatrixCSC,
                                     scale_name::String)
    results = InferenceResult[]
    n_users, n_items = size(X)

    # recommend(model, X; k=10)
    r = bench(() -> recommend(model, X; k=10))
    push!(results, InferenceResult(
        scale_name, algo_name, "recommend(k=10)",
        r.time, r.mem, n_users, n_items, n_users / r.time))

    # recommend(model, X; k=50)
    r = bench(() -> recommend(model, X; k=50))
    push!(results, InferenceResult(
        scale_name, algo_name, "recommend(k=50)",
        r.time, r.mem, n_users, n_items, n_users / r.time))

    # score(model, X) — skip if output would exceed 2 GiB
    score_size = _score_bytes(n_users, n_items)
    if score_size < 2 * 1024^3
        r = bench(() -> score(model, X))
        push!(results, InferenceResult(
            scale_name, algo_name, "score(full)",
            r.time, r.mem, n_users, n_items, n_users / r.time))
    else
        push!(results, InferenceResult(
            scale_name, algo_name, "score(full)",
            NaN, 0, n_users, n_items, NaN))
    end

    # similar_items — only for MF models that support it
    if model isa AbstractMatrixFactorization
        n_queries = min(1000, n_items)
        r = bench(() -> begin
            for i in 1:n_queries
                similar_items(model, i; k=10)
            end
        end)
        push!(results, InferenceResult(
            scale_name, algo_name, "similar_items(×$(n_queries))",
            r.time, r.mem, n_users, n_items, n_queries / r.time))
    end

    return results
end

function run_regression_benchmarks(model, algo_name::String,
                                    X::SparseMatrixCSC, scale_name::String)
    results = InferenceResult[]
    n_samples = size(X, 1)
    n_features = size(X, 2)

    r = bench(() -> predict(model, X))
    push!(results, InferenceResult(
        scale_name, algo_name, "predict",
        r.time, r.mem, n_samples, n_features, n_samples / r.time))

    return results
end

function main()
    println("=" ^ 78)
    println("Gideon.jl — Inference (recommend / score / predict) Benchmark")
    println("=" ^ 78)
    println("  Julia:        $(VERSION)")
    println("  Threads:      $(Threads.nthreads())")
    println("  BLAS threads: $(BLAS.get_num_threads())")
    println("  Element type: Float32")
    println("  Peak RSS:     $(_fmt_bytes(_peak_rss_bytes()))")
    println()

    all_results = InferenceResult[]

    # ── Recommender models (MF + item-similarity) ──
    for scale in SCALES
        println("━" ^ 78)
        @printf("Scale: %s (%d users × %d items, density=%.3f)\n",
                scale.name, scale.n_users, scale.n_items, scale.density)
        println("━" ^ 78)

        X = generate_matrix(scale.n_users, scale.n_items, scale.density)
        @printf("  Matrix: %d users × %d items, nnz=%d (%.1f MiB sparse)\n\n",
                size(X,1), size(X,2), nnz(X),
                (nnz(X) * (4+8) + (scale.n_items+1)*8) / 1024^2)

        for (algo_name, model_fn) in vcat(mf_models(), similarity_models())
            if skip_at_scale(algo_name, scale.n_items)
                @printf("  %-12s  SKIPPED (too slow at n_items=%d)\n", algo_name, scale.n_items)
                continue
            end

            @printf("  %-12s  training... ", algo_name)
            flush(stdout)
            model = model_fn()
            GC.gc()
            t_train = @elapsed fit!(model, X; rng=MersenneTwister(42))
            @printf("%.2fs  →  benchmarking inference...\n", t_train)

            results = run_recommender_benchmarks(model, algo_name, X, scale.name)
            append!(all_results, results)

            for r in results
                if isnan(r.time_seconds)
                    @printf("    %-25s  SKIPPED (>2 GiB output)\n", r.operation)
                else
                    @printf("    %-25s  %8.4f s  %10s  %10.0f users/s\n",
                            r.operation, r.time_seconds,
                            _fmt_bytes(r.memory_bytes), r.throughput)
                end
            end
            println()

            # Free model memory between algorithms
            model = nothing
            GC.gc()
        end

        # Free matrix between scales
        X = nothing
        GC.gc()
    end

    # ── Regression models ──
    println("\n" * "━" ^ 78)
    println("Regression models (FTRL, FM)")
    println("━" ^ 78)

    for (n_samples, n_features, density) in [
        (10_000,  500,   0.05),
        (100_000, 1_000, 0.02),
    ]
        scale_name = "$(n_samples÷1000)k×$(n_features)"
        X_reg, y_reg = generate_regression_data(n_samples, n_features, density)
        @printf("  Data: %d samples × %d features, nnz=%d\n\n",
                n_samples, n_features, nnz(X_reg))

        for (algo_name, model_fn) in regression_models()
            @printf("  %-12s  training... ", algo_name)
            flush(stdout)
            model = model_fn()
            GC.gc()
            t_train = @elapsed fit!(model, X_reg, y_reg; rng=MersenneTwister(42))
            @printf("%.2fs  →  benchmarking predict...\n", t_train)

            results = run_regression_benchmarks(model, algo_name, X_reg, scale_name)
            append!(all_results, results)

            for r in results
                @printf("    %-25s  %8.4f s  %10s  %10.0f samples/s\n",
                        r.operation, r.time_seconds,
                        _fmt_bytes(r.memory_bytes), r.throughput)
            end
            println()

            model = nothing
            GC.gc()
        end
    end

    # ── Save CSV ──
    outpath = joinpath(@__DIR__, "results_recommend_julia.csv")
    open(outpath, "w") do io
        println(io, "scale,algorithm,operation,time_seconds,memory_bytes," *
                    "n_users,n_items,throughput")
        for r in all_results
            @printf(io, "%s,%s,%s,%.6f,%d,%d,%d,%.1f\n",
                    r.scale, r.algorithm, r.operation,
                    r.time_seconds, r.memory_bytes,
                    r.n_users, r.n_items, r.throughput)
        end
    end

    # ── Summary table ──
    println("\n" * "=" ^ 78)
    println("SUMMARY")
    println("=" ^ 78)
    println("  Peak RSS: $(_fmt_bytes(_peak_rss_bytes()))")
    println()
    @printf("%-8s %-12s %-25s %10s %10s %12s\n",
            "Scale", "Algorithm", "Operation", "Time (s)", "Memory", "Throughput")
    println("─" ^ 78)
    for r in all_results
        if isnan(r.time_seconds)
            @printf("%-8s %-12s %-25s %10s %10s %12s\n",
                    r.scale, r.algorithm, r.operation, "SKIP", "-", "-")
        else
            @printf("%-8s %-12s %-25s %10.4f %10s %10.0f /s\n",
                    r.scale, r.algorithm, r.operation,
                    r.time_seconds, _fmt_bytes(r.memory_bytes), r.throughput)
        end
    end
    println("─" ^ 78)
    println("\nResults saved to: $outpath")
end

main()
