# benchmark/compare.jl — Compare benchmark runs.
#
# Loads benchmark/logs/results.jsonl and compares the most recent run of each
# (algorithm, scale, config) against the previous run made from a different
# git commit. Prints a delta table and optionally exits non-zero when a
# regression exceeds the threshold.
#
# Usage:
#   julia --project=. benchmark/compare.jl                # print table
#   julia --project=. benchmark/compare.jl --strict       # exit 1 on regression
#   julia --project=. benchmark/compare.jl --threshold 15 # custom % threshold

using JSON
using Printf
using Dates

const LOG_FILE = joinpath(@__DIR__, "logs", "results.jsonl")

struct Run
    git_sha::String
    date::DateTime
    algorithm::String
    config::String
    scale::String
    nthreads::Int
    blas_threads::Int
    fit_seconds::Float64
    fit_bytes::Float64
    recommend_seconds::Float64
    recommend_bytes::Float64
end

function parse_run(record::Dict)
    Run(
        record["git_sha"],
        DateTime(record["date"]),
        record["algorithm"],
        get(record, "config", ""),
        record["scale"],
        get(record, "nthreads", 1),
        get(record, "blas_threads", 1),
        Float64(record["fit_seconds"]),
        Float64(get(record, "fit_bytes", 0.0)),
        Float64(record["recommend_seconds"]),
        Float64(get(record, "recommend_bytes", 0.0)),
    )
end

function load_runs()
    isfile(LOG_FILE) || error("$LOG_FILE not found — run benchmark/run.jl first")
    runs = Run[]
    for line in eachline(LOG_FILE)
        isempty(strip(line)) && continue
        push!(runs, parse_run(JSON.parse(line)))
    end
    runs
end

function main()
    threshold = 10.0
    strict = false
    for arg in ARGS
        if arg == "--strict"
            strict = true
        elseif startswith(arg, "--threshold=")
            threshold = parse(Float64, split(arg, '=')[2])
        end
    end

    runs = load_runs()
    isempty(runs) && (println("No benchmark records found."); return)

    # Most recent run per (algorithm, scale, config, environment) = current.
    # Previous run with a different git sha = baseline.
    by_key = Dict{Tuple{String,String,String,Int,Int},Vector{Run}}()
    for r in runs
        key = (r.algorithm, r.scale, r.config, r.nthreads, r.blas_threads)
        push!(get!(by_key, key, Run[]), r)
    end

    println(@sprintf("%-13s %-10s %-9s %8s %8s %7s %9s", "algorithm", "scale", "sha", "fit (s)", "base (s)", "delt%", "status"))
    println("-" ^ 78)

    regressions = String[]
    nrows = 0
    for key in sort(collect(keys(by_key)))
        entries = sort(by_key[key]; by=r -> r.date)
        current = entries[end]
        baseline = nothing
        for r in reverse(entries[1:end-1])
            if r.git_sha != current.git_sha
                baseline = r
                break
            end
        end

        nrows += 1
        if baseline === nothing
            println(@sprintf("%-13s %-10s %-9s %8.3f %8s %7s %9s",
                current.algorithm, current.scale, current.git_sha,
                current.fit_seconds, "—", "—", "no base"))
            continue
        end

        delta = (current.fit_seconds - baseline.fit_seconds) / baseline.fit_seconds * 100
        status = delta > threshold ? "REGRESSED" : (delta < -threshold ? "improved" : "ok")
        delta > threshold && push!(regressions,
            "$(current.algorithm)/$(current.scale): +$(round(delta; digits=1))% " *
            "($(round(baseline.fit_seconds; digits=3))s → $(round(current.fit_seconds; digits=3))s, " *
            "base $(baseline.git_sha) vs $(current.git_sha))")
        println(@sprintf("%-13s %-10s %-9s %8.3f %8.3f %7.1f %9s",
            current.algorithm, current.scale, current.git_sha,
            current.fit_seconds, baseline.fit_seconds, delta, status))
    end

    println("-" ^ 78)
    println("$nrows (algorithm, scale, config) groups; threshold: $threshold%")

    if !isempty(regressions)
        println("\nRegressions (> $threshold%):")
        foreach(r -> println("  ✗ $r"), regressions)
        if strict
            exit(1)
        end
    else
        println("\nNo regressions detected.")
    end
end

main()
