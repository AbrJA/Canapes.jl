#!/usr/bin/env julia
# validation/run.jl — One-command runner for R / Python reference validation.
#
# Usage:
#   julia --project=. validation/run.jl --all            # prepare fixtures + run both
#   julia --project=. validation/run.jl --r              # R comparison only
#   julia --project=. validation/run.jl --python         # Python comparison only
#   julia --project=. validation/run.jl --prepare        # prepare both fixture sets
#   julia --project=. validation/run.jl                  # run both (fixtures must exist)
#   julia --project=. validation/run.jl --help
#
# Each step runs in its own subprocess: exit codes propagate, and a failure in
# one comparison does not prevent the others from running. The script exits
# non-zero if any requested step fails or is skipped because its tooling
# (`uvr`/`Rscript`, `python3`, or the Python `implicit` package) is missing.
# R fixture generation uses `uvr run` (managed R env from uvr.toml) when
# available, falling back to a system `Rscript`.

const ROOT = normpath(joinpath(@__DIR__, ".."))
cd(ROOT)

const FLAGS = Set(ARGS)
_has(f::String) = f in FLAGS

const NO_ARGS = isempty(ARGS)
const ALL     = _has("--all")
const PREP    = _has("--prepare") || ALL

# `--prepare` alone implies both fixture sets; `--r`/`--python` narrow it.
const RUN_R   = _has("--r") || ALL || (NO_ARGS && !PREP)
const RUN_PY  = _has("--python") || ALL || (NO_ARGS && !PREP)
const PREP_R  = PREP && (_has("--r") || !(_has("--r") || _has("--python")))
const PREP_PY = PREP && (_has("--python") || !(_has("--r") || _has("--python")))

if _has("--help")
    println("""
    Gideon reference validation runner

    Flags:
      --all        Prepare fixtures and run both R and Python comparisons
      --r          Run only the R (rsparse) comparison
      --python     Run only the Python (implicit/sklearn/scipy) comparison
      --prepare    Generate fixtures (implies --r --python unless --r/--python given)
      --help       Show this help

    Fixture directories (overridable):
      GIDEON_R_FIXTURE_DIR  (default /tmp/gideon_fixtures)
      GIDEON_PY_FIXTURE_DIR (default /tmp/gideon_fixtures/python)

    Exit code is 0 only if every requested step passed.
    """)
    exit(0)
end

println("\nGideon reference validation runner")
println("  R comparison:      $RUN_R")
println("  Python comparison: $RUN_PY")
println("  prepare R:         $PREP_R")
println("  prepare Python:    $PREP_PY")

# (label, ok, skipped) per step, for the summary table
results = Tuple{String,Bool,Bool}[]

function run_step(cmd::Cmd, label::String)
    println("\n━━━ $label ━━━")
    flush(stdout)
    ok = success(pipeline(cmd; stdout=stdout, stderr=stderr))
    println(ok ? "✓ $label passed" : "✗ $label FAILED")
    push!(results, (label, ok, false))
    return ok
end

function skip_step(label::String, reason::String)
    println("✗ $label skipped: $reason")
    push!(results, (label, false, true))
    return false
end

const JULIA = `$(Base.julia_cmd()) --project=$ROOT`

# ── Fixture preparation ───────────────────────────────────────────────────────

if PREP_R
    # Prefer uvr (managed R env, see uvr.toml); fall back to a system Rscript.
    uvr_r_cmd = if isnothing(Sys.which("uvr"))
        nothing
    else
        `uvr run validation/fixtures_r.R`
    end
    if !isnothing(uvr_r_cmd)
        run_step(uvr_r_cmd, "Prepare R fixtures (uvr)")
    elseif isnothing(Sys.which("Rscript"))
        skip_step("Prepare R fixtures",
                  "neither `uvr` nor `Rscript` found (install uvr, or R with the rsparse package)")
    else
        run_step(`Rscript validation/fixtures_r.R`, "Prepare R fixtures")
    end
end
if PREP_PY
    py = get(ENV, "PYTHON", "python3")
    if isnothing(Sys.which(py))
        skip_step("Prepare Python fixtures", "Python executable not found: $py")
    else
        run_step(`$py validation/fixtures_py.py`, "Prepare Python fixtures")
    end
end

# ── Reference comparisons ─────────────────────────────────────────────────────

RUN_R  && run_step(`$JULIA validation/validate_r.jl`,  "R reference comparison")
RUN_PY && run_step(`$JULIA validation/validate_py.jl`, "Python reference comparison")

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n" * "─"^60)
println("Validation summary")
if isempty(results)
    println("  (no steps ran — nothing was requested)")
    println("─"^60)
    println("OVERALL: FAILED")
    exit(1)
end
for (label, ok, skipped) in results
    status = skipped ? "SKIPPED" : (ok ? "PASSED" : "FAILED")
    println("  $(rpad(label, 32)) $status")
end
println("─"^60)

all_ok = all(r -> r[2], results)
println(all_ok ? "OVERALL: PASSED" : "OVERALL: FAILED")
exit(all_ok ? 0 : 1)
