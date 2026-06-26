# benchmark/compare_recommend.jl
# Compare Julia vs Python inference benchmark results.
#
# Run after both benchmarks have completed:
#   julia benchmark/compare_recommend.jl

using Printf

function load_csv(path)
    rows = Dict{String, Any}[]
    lines = readlines(path)
    isempty(lines) && return rows
    headers = split(lines[1], ',')
    for line in lines[2:end]
        fields = split(line, ',')
        row = Dict{String, Any}()
        for (h, f) in zip(headers, fields)
            h = strip(String(h))
            f = strip(String(f))
            if h in ("time_seconds", "throughput")
                row[h] = tryparse(Float64, f)
            elseif h in ("memory_bytes", "n_users", "n_items")
                row[h] = tryparse(Int, f)
            else
                row[h] = f
            end
        end
        push!(rows, row)
    end
    return rows
end

function fmt_bytes(b)
    b === nothing && return "-"
    b < 1024       && return @sprintf("%d B", b)
    b < 1024^2     && return @sprintf("%.1f KiB", b / 1024)
    b < 1024^3     && return @sprintf("%.1f MiB", b / 1024^2)
    return @sprintf("%.1f GiB", b / 1024^3)
end

function main()
    dir = @__DIR__
    jl_path = joinpath(dir, "results_recommend_julia.csv")
    py_path = joinpath(dir, "results_recommend_python.csv")

    if !isfile(jl_path)
        println("Missing: $jl_path")
        println("Run: julia --project -t8 benchmark/bench_recommend.jl")
        return
    end
    if !isfile(py_path)
        println("Missing: $py_path")
        println("Run: python benchmark/bench_recommend.py")
        return
    end

    jl = load_csv(jl_path)
    py = load_csv(py_path)

    # Build lookup: (scale, operation) → algorithm mapping
    # Match algorithms across languages
    ALGO_MAP = Dict(
        # Julia name => Python name
        "WRMF-Chol" => "ALS(implicit)",
        "WRMF-CG"   => "ALS(implicit)",
        "iALS"       => "ALS(implicit)",
        "BPR"        => "BPR(implicit)",
        "LMF"        => "LMF(implicit)",
        "EASE"       => "EASE",
        "ItemKNN"    => "ItemKNN",
    )

    # Index Python results
    py_idx = Dict{Tuple{String,String,String}, Dict{String,Any}}()
    for r in py
        key = (r["scale"], r["algorithm"], r["operation"])
        py_idx[key] = r
    end

    println("=" ^ 90)
    println("Julia vs Python — Inference Benchmark Comparison")
    println("=" ^ 90)
    @printf("%-8s %-12s %-20s │ %10s %10s │ %7s\n",
            "Scale", "Algorithm", "Operation", "Julia (s)", "Python (s)", "Speedup")
    println("─" ^ 90)

    for r in jl
        jl_algo = r["algorithm"]
        py_algo = get(ALGO_MAP, jl_algo, nothing)
        py_algo === nothing && continue

        key = (r["scale"], py_algo, r["operation"])
        py_r = get(py_idx, key, nothing)
        py_r === nothing && continue

        jl_t = r["time_seconds"]
        py_t = py_r["time_seconds"]
        (jl_t === nothing || py_t === nothing) && continue
        (isnan(jl_t) || isnan(py_t)) && continue

        speedup = py_t / jl_t
        marker = speedup >= 1.0 ? "✓" : "✗"

        @printf("%-8s %-12s %-20s │ %10.4f %10.4f │ %6.1f× %s\n",
                r["scale"], jl_algo, r["operation"],
                jl_t, py_t, speedup, marker)
    end

    println("─" ^ 90)
    println("\n✓ = Julia faster, ✗ = Python faster")

    # Memory comparison
    println("\n" * "=" ^ 90)
    println("Memory Usage Comparison (RSS delta)")
    println("=" ^ 90)
    @printf("%-8s %-12s %-20s │ %12s %12s\n",
            "Scale", "Algorithm", "Operation", "Julia", "Python")
    println("─" ^ 90)

    for r in jl
        jl_algo = r["algorithm"]
        py_algo = get(ALGO_MAP, jl_algo, nothing)
        py_algo === nothing && continue

        key = (r["scale"], py_algo, r["operation"])
        py_r = get(py_idx, key, nothing)
        py_r === nothing && continue

        jl_m = r["memory_bytes"]
        py_m = py_r["memory_bytes"]
        (jl_m === nothing || py_m === nothing) && continue

        @printf("%-8s %-12s %-20s │ %12s %12s\n",
                r["scale"], jl_algo, r["operation"],
                fmt_bytes(jl_m), fmt_bytes(py_m))
    end
    println("─" ^ 90)
end

main()
