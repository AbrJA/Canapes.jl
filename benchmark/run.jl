# benchmark/run.jl — Tracked performance harness.
#
# Runs every algorithm at three fixed scales, records fit!/recommend time,
# allocations and heap growth, and appends one JSONL record per result to
# benchmark/logs/results.jsonl (git sha, environment, config and metrics).
#
# Usage:
#   julia --project=benchmark --threads=8 benchmark/run.jl
#
# Compare runs:
#   julia --project=benchmark benchmark/compare.jl --strict
#
# Benchmarks are deterministic: fixed seeds, fixed matrices. Run on the same
# machine and thread count to compare commits.

using Canapes
using SparseArrays
using Random
using LinearAlgebra
using Printf
using Dates
using JSON

const LOG_DIR = joinpath(@__DIR__, "logs")
const LOG_FILE = joinpath(LOG_DIR, "results.jsonl")
const K = 10

# Optional scale filter: BENCH_SCALE=hundreds|thousands|millions runs a single
# scale (useful to smoke-test the smallest scale before the long ones).
const SCALE_FILTER = get(ENV, "BENCH_SCALE", "")

const SCALES = [
    (name="hundreds",  n_users=500,     n_items=300,      density=0.10),  # ~15K nnz
    (name="thousands", n_users=5_000,   n_items=3_000,    density=0.02),  # ~300K nnz
    (name="millions",  n_users=100_000, n_items=50_000,   density=0.001), # ~5M nnz
]

# ──────────────────────────────────────────────────────────────────────────────
# Deterministic data generation
# ──────────────────────────────────────────────────────────────────────────────

function generate_matrix(n_users::Int, n_items::Int, density::Float64; seed::Int=42)
    rng = MersenneTwister(seed)
    nnz_target = round(Int, n_users * n_items * density)
    rows = rand(rng, 1:n_users, nnz_target)
    cols = rand(rng, 1:n_items, nnz_target)
    sparse(rows, cols, ones(Float64, nnz_target), n_users, n_items)
end

function generate_cooccurrence(n::Int, density::Float64; seed::Int=7)
    rng = MersenneTwister(seed)
    A = sprand(rng, n, n, density)
    A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
    A
end

# ──────────────────────────────────────────────────────────────────────────────
# Measurement
# ──────────────────────────────────────────────────────────────────────────────

function measure(f)
    live0 = Base.gc_live_bytes()
    t0 = time_ns()
    bytes = @allocated f()
    t1 = time_ns()
    (
        seconds = (t1 - t0) / 1e9,
        bytes = bytes,
        live_bytes = Base.gc_live_bytes(),
    )
end

function best_of(f, reps::Int)
    best = measure(f)
    for _ in 2:reps
        m = measure(f)
        m.seconds < best.seconds && (best = m)
    end
    best
end

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark definitions
# ──────────────────────────────────────────────────────────────────────────────

# Each spec is (name, model, fit!, infer, dense, infer_millions).
#   fit!(model, X, rng) → nothing
#   infer(model, X)      → output (recommendations, predictions or scores)
#   dense::Bool          — output is a dense n_users × n_items matrix (skip
#                          at the "millions" scale, where it would be 100K×50K)
#   infer_millions       — optional scale-scaled inference for dense-output
#                          models: pairwise score over `pairs` (sampled 1M
#                          (user, item) pairs) instead of the dense matrix.
#                          Models without an infer_millions entry are skipped
#                          at the millions scale.
# At millions, only models that fit AND infer in < 15 s are tracked (verified
# per-model with a probe); SoftImpute (23 s) and LogisticMF (40 s) are excluded
# for that reason, as are the O(n_items²) item-item fits.
function benchmark_specs(X, pairs)
    y = rand(MersenneTwister(3), size(X, 1))
    pu, pi = pairs
    [
        # ── implicit ranking (recommend) ──
        (name="WeightedMF-Cholesky", dense=false,
         model=WeightedMF(rank=64, λ=0.1, α=40.0, max_iter=10, solver=CholeskySolver(), verbose=false),
         fit=(m,r)->fit!(m, X; rng=r), infer=m->recommend(m, X; k=K)),
        (name="WeightedMF-CG", dense=false,
         model=WeightedMF(rank=64, λ=0.1, α=40.0, max_iter=10, solver=CGSolver(), cg_steps=3, verbose=false),
         fit=(m,r)->fit!(m, X; rng=r), infer=m->recommend(m, X; k=K)),
        (name="CachedALS", dense=false,
         model=CachedALS(rank=64, max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X; rng=r), infer=m->recommend(m, X; k=K)),
        (name="ElementwiseALS", dense=false,
         model=ElementwiseALS(rank=64, max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X; rng=r), infer=m->recommend(m, X; k=K)),
        (name="PairwiseRanking", dense=false,
         model=PairwiseRanking(rank=64, λ_user=0.01, λ_pos=0.01, λ_neg=0.01, lr=0.05, max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X; rng=r), infer=m->recommend(m, X; k=K)),

        # ── item similarity (recommend) ──
        (name="ShallowAutoencoder", dense=true,  # O(n_items²)
         model=ShallowAutoencoder(λ=200.0, verbose=false),
         fit=(m,r)->fit!(m, X), infer=m->recommend(m, X; k=K)),
        (name="SparseLinearModel", dense=true,  # O(n_items²)
         model=SparseLinearModel(max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X), infer=m->recommend(m, X; k=K)),
        (name="SparseLinearADMM", dense=true,  # O(n_items²)
         model=SparseLinearADMM(max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X), infer=m->recommend(m, X; k=K)),
        (name="ItemKNN", dense=false,  # memory-bounded column-by-column top-k; probed < 15 s at millions
         model=ItemKNN(k=50, verbose=false),
         fit=(m,r)->fit!(m, X), infer=m->recommend(m, X; k=K)),
        (name="GraphRandomWalk", dense=false,  # sparse W; probed ~10 s fit at millions
         model=GraphRandomWalk(β=0.0, k=50, verbose=false),
         fit=(m,r)->fit!(m, X), infer=m->recommend(m, X; k=K)),

        # ── explicit rating prediction (predict — dense output) ──
        (name="BiasedMF", dense=true,
         model=WeightedMF(rank=64, λ=0.1, max_iter=10, solver=CholeskySolver(), feedback=Explicit, verbose=false),
         fit=(m,r)->fit!(m, X; rng=r), infer=m->predict(m, X),
         infer_millions=m->score(m, pu, pi)),
        (name="BaselineOnly", dense=true,
         model=BaselineOnly(λ=0.02, max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X), infer=m->predict(m, X),
         infer_millions=m->score(m, pu, pi)),
        (name="SlopeOne", dense=true,  # O(n_items²) fit — millions skipped
         model=SlopeOne(verbose=false),
         fit=(m,r)->fit!(m, X), infer=m->predict(m, X)),
        (name="PearsonKNN", dense=true,  # O(n_users²) fit — millions skipped
         model=PearsonKNN(k=50, verbose=false),
         fit=(m,r)->fit!(m, X), infer=m->predict(m, X)),

        # ── completion (score — dense output) ──
        (name="SoftImpute", dense=true,  # 23 s fit at millions — skipped (gate < 15 s)
         model=SoftImpute(rank=32, λ=0.5, max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X; rng=r), infer=m->score(m, X)),
        (name="SoftSVD", dense=true,
         model=SoftSVD(rank=32, max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X; rng=r), infer=m->score(m, X),
         infer_millions=m->score(m, pu, pi)),
        (name="PureSVD", dense=true,
         model=PureSVD(rank=32, max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X; rng=r), infer=m->score(m, X),
         infer_millions=m->score(m, pu, pi)),

        # ── sparse regression (predict — vector output) ──
        (name="FTRL", dense=false,
         model=FTRL(lr=0.1, max_iter=1, verbose=false),
         fit=(m,r)->update!(m, X, y; rng=r), infer=m->predict(m, X)),
        (name="FactorizationMachine", dense=false,
         model=FactorizationMachine(rank=8, max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X, y; rng=r), infer=m->predict(m, X)),

        # ── experimental ──
        (name="ProbabilisticMF", dense=true,
         model=Canapes.Experimental.ProbabilisticMF(rank=32, λ=0.1, lr=0.05, max_iter=10, verbose=false),
         fit=(m,r)->fit!(m, X; rng=r), infer=m->predict(m, X),
         infer_millions=m->score(m, pu, pi)),
    ]
end

function compile_warmup()
    X = sparse([1, 2, 3], [1, 2, 1], [1.0, 1.0, 1.0], 50, 40)
    y = rand(MersenneTwister(3), size(X, 1))
    for spec in benchmark_specs(X, (Int[1, 2], Int[1, 2]))
        spec.fit(spec.model, MersenneTwister(1))
        spec.infer(spec.model)
        if get(spec, :infer_millions, nothing) !== nothing
            spec.infer_millions(spec.model)
        end
    end
    C = sprand(40, 40, 0.2)
    C = C + C'
    nonzeros(C) .= abs.(nonzeros(C)) .+ 0.1
    g = GlobalVectors(rank=8, max_iter=2, verbose=false)
    fit!(g, C; rng=MersenneTwister(1))
    recommend(g, C; k=K)
end

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

function git_sha()
    try
        strip(String(read(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`)))
    catch
        "unknown"
    end
end

function config_string(model)
    parts = String[]
    for f in fieldnames(typeof(model))
        v = getfield(model, f)
        if !(v isa AbstractArray) && f != :is_fitted
            push!(parts, string(f, "=", v))
        end
    end
    join(parts, ", ")
end

function log_record(record)
    mkpath(LOG_DIR)
    open(LOG_FILE; append=true) do io
        println(io, JSON.json(record))
    end
    record
end

function print_record(record)
    @printf("  %-13s %-10s fit=%8.3fs  alloc=%9.1fMB  rec=%6.3fs  alloc=%6.1fMB\n",
        record["algorithm"], record["scale"],
        record["fit_seconds"], record["fit_bytes"] / 2^20,
        record["recommend_seconds"], record["recommend_bytes"] / 2^20)
    flush(stdout)
end

function env_record()
    Dict(
        "git_sha" => git_sha(),
        "julia_version" => string(VERSION),
        "date" => string(now()),
        "nthreads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 78)
    println("Canapes.jl performance harness")
    println("=" ^ 78)
    println("  Julia:     $(VERSION)")
    println("  Threads:   $(Threads.nthreads())")
    println("  BLAS:      $(BLAS.get_num_threads()) threads")
    println("  Git:       $(git_sha())")
    println("  Log:       $LOG_FILE")
    println()

    println("Compile warmup...")
    compile_warmup()

    reps = Dict("hundreds" => 3, "thousands" => 2, "millions" => 1)

    for scale in SCALES
        isempty(SCALE_FILTER) || scale.name == SCALE_FILTER || continue
        println("Scale: $(scale.name)  ($(scale.n_users) users × $(scale.n_items) items, " *
                "density $(scale.density))")
        flush(stdout)
        X = generate_matrix(scale.n_users, scale.n_items, scale.density)
        @printf("  X nnz: %d\n\n", nnz(X))

        # Dense-output models (explicit predict / completion score) build a full
        # n_users × n_items matrix — at millions they run `infer_millions`
        # (pairwise score over 1M sampled pairs) when available, else skipped.
        if scale.name == "millions"
            rngp = MersenneTwister(99)
            pairs = (rand(rngp, 1:scale.n_users, 1_000_000),
                     rand(rngp, 1:scale.n_items, 1_000_000))
        else
            pairs = (Int[], Int[])
        end
        for spec in benchmark_specs(X, pairs)
            infer = (spec.dense && scale.name == "millions") ?
                    get(spec, :infer_millions, nothing) : spec.infer
            infer === nothing && continue           # no scale-scaled inference available
            model = spec.model
            cfg = config_string(model)

            fit_m = best_of(() -> spec.fit(model, MersenneTwister(1)), reps[scale.name])
            rec_m = best_of(() -> infer(model), reps[scale.name])

            record = log_record(merge(env_record(), Dict(
                "algorithm" => spec.name,
                "config" => cfg,
                "scale" => scale.name,
                "n_users" => scale.n_users,
                "n_items" => scale.n_items,
                "nnz" => nnz(X),
                "fit_seconds" => fit_m.seconds,
                "fit_bytes" => fit_m.bytes,
                "fit_live_bytes" => fit_m.live_bytes,
                "recommend_seconds" => rec_m.seconds,
                "recommend_bytes" => rec_m.bytes,
                "recommend_live_bytes" => rec_m.live_bytes,
            )))
            print_record(record)
        end

        # GlobalVectors runs on a square co-occurrence matrix sized to match the scale.
        n = scale.n_users
        d = scale.density * scale.n_items / n
        C = generate_cooccurrence(n, d)
        glove = GlobalVectors(rank=64, lr=0.05, max_iter=10, verbose=false)
        fit_m = best_of(() -> fit!(glove, C; rng=MersenneTwister(1)), reps[scale.name])
        rec_m = best_of(() -> recommend(glove, C; k=K), reps[scale.name])
        record = log_record(merge(env_record(), Dict(
            "algorithm" => "GlobalVectors",
            "config" => "rank=64,lr=0.05,max_iter=10",
            "scale" => scale.name,
            "n_users" => n,
            "n_items" => n,
            "nnz" => nnz(C),
            "fit_seconds" => fit_m.seconds,
            "fit_bytes" => fit_m.bytes,
            "fit_live_bytes" => fit_m.live_bytes,
            "recommend_seconds" => rec_m.seconds,
            "recommend_bytes" => rec_m.bytes,
            "recommend_live_bytes" => rec_m.live_bytes,
        )))
        print_record(record)
        println()
    end

    println("Done. Results appended to $LOG_FILE")
end

main()
