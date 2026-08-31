using Documenter, Gideon

makedocs(
    modules  = [Gideon],
    sitename = "Gideon.jl",
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
    repo = "github.com/AbrJA/Gideon.jl.git",
    push_preview = true,
)
