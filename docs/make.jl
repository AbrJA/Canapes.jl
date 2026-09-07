using Documenter, Canapes

makedocs(
    modules  = [Canapes],
    sitename = "Canapes.jl",
    format   = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    doctest  = true,
    checkdocs = :exports,
    pages    = [
        "Home"       => "index.md",
        "Algorithms" => "algorithms.md",
        "Metrics"    => "metrics.md",
        "API"        => "api.md",
    ],
)

deploydocs(
    repo = "github.com/AbrJA/Canapes.jl.git",
    push_preview = true,
)
