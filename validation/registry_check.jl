# validation/registry_check.jl — package-side registry-readiness validation
#
# Enforces the General-registry rules for a new package (the checks
# RegistryCI.jl applies to registration PRs) so they run on every PR to this
# repo instead of only at registration time:
#   - valid package name, UUID and semver version
#   - [compat] entry for every non-stdlib dependency and weak dependency
#   - every compat bound parses as a VersionSpec (incl. the `julia` bound)
#   - no stale dependencies (non-stdlib deps unused in src/)
#
# Exit code is non-zero on any violation.
# Run with: julia --project=. validation/registry_check.jl

using Pkg, UUIDs

project_path = joinpath(@__DIR__, "..", "Project.toml")
project = Pkg.TOML.parsefile(project_path)

function check(cond::Bool, msg::AbstractString)
    cond || error("Registry check failed: $msg")
    nothing
end

# ── name / uuid / version ─────────────────────────────────────────────────────
name = project["name"]
check(Base.isidentifier(name), "package name is not a valid identifier: '$name'")
uuid = tryparse(UUID, string(project["uuid"]))
check(uuid !== nothing, "invalid package uuid: $(project["uuid"])")
version = tryparse(VersionNumber, string(project["version"]))
check(version !== nothing, "invalid package version: $(project["version"])")
check(version.major >= 0 && version.prerelease === (), "version must be a stable semver, got $version")
println("✓ name=$(name), uuid=$(uuid), version=$version")

# ── compat coverage ───────────────────────────────────────────────────────────
stdlibs = Set{String}(name for (_, (name, _)) in Pkg.Types.stdlibs())
compat = get(project, "compat", Dict{String,Any}())

check(haskey(compat, "julia"), "missing [compat] entry for `julia`")
try
    Pkg.Versions.VersionSpec(compat["julia"])
catch
    check(false, "invalid `julia` compat bound: $(compat["julia"])")
end

for section in ("deps", "weakdeps")
    for (dep, _) in get(project, section, Dict{String,Any}())
        dep in stdlibs && continue
        check(haskey(compat, dep),
              "missing [compat] entry for $section entry '$dep'")
        try
            Pkg.Versions.VersionSpec(compat[dep])
        catch
            check(false, "invalid compat bound for '$dep': $(compat[dep])")
        end
    end
end
println("✓ compat entries present and valid for all non-stdlib deps/weakdeps")

# ── stale dependencies ────────────────────────────────────────────────────────
src_dir = joinpath(@__DIR__, "..", "src")
src_files = filter(isfile, joinpath.(src_dir, readdir(src_dir; join=true)))
src_text = join(read(f, String) for f in src_files)
for (dep, _) in get(project, "deps", Dict{String,Any}())
    dep in stdlibs && continue
    check(occursin(dep, src_text),
          "stale dependency: '$dep' is declared but never used in src/")
end
println("✓ no stale non-stdlib dependencies")

println("All registry-readiness checks passed.")
