# test/test_docs_examples.jl — dynamically extract and run every ```julia block
# from docs/src/*.md, so the docs snippets cannot drift from the API (the
# Documenter build only executes @example blocks, so plain ```julia snippets
# were previously never run). Setup blocks (Pkg.add) are skipped explicitly.

@testset "Docs examples" begin
    docsdir = joinpath(dirname(@__DIR__), "docs", "src")
    md_files = sort(filter(f -> endswith(f, ".md"), readdir(docsdir)))

    # Extract ```julia ... ``` fenced blocks
    blocks = String[]
    sources = String[]
    for f in md_files
        text = read(joinpath(docsdir, f), String)
        text = replace(text, "\r\n" => "\n")
        for m in eachmatch(r"```julia\n(.*?)```"s, text)
            push!(blocks, m.captures[1])
            push!(sources, f)
        end
    end
    @test !isempty(blocks)

    failed = String[]
    skipped = String[]
    for (i, code) in enumerate(blocks)
        if occursin("Pkg.add", code)
            push!(skipped, "$(sources[i]) block $i (setup)")
            continue
        end
        try
            include_string(Main, code)
        catch e
            push!(failed, "$(sources[i]) block $i: $(sprint(showerror, e))")
        end
    end

    @test isempty(failed)
    if !isempty(failed)
        println("\nDocs example failures:")
        foreach(f -> println("  ✗ ", f), failed)
    end
    @test length(skipped) >= 1  # the installation block is expected to be skipped
end