# test/test_readme.jl — dynamically extract and run every ```julia block from
# README.md (like validate_docs.jl does for docs/src). Blocks that need
# unavailable dependencies (CUDA) are skipped explicitly.

@testset "README examples" begin
    readme = read(joinpath(dirname(@__DIR__), "README.md"), String)

    # Extract ```julia ... ``` fenced blocks
    blocks = String[]
    for m in eachmatch(r"```julia\n(.*?)```"s, readme)
        push!(blocks, m.captures[1])
    end
    @test !isempty(blocks)

    failed = String[]
    skipped = String[]
    for (i, code) in enumerate(blocks)
        # GPU examples require CUDA and use pseudo-code signatures; the
        # Installation block calls Pkg.add — neither is meant to execute here.
        if occursin("using Canapes, CUDA", code) || occursin("fit_gpu!", code) ||
           occursin("Pkg.add", code)
            push!(skipped, "block $i (setup/GPU)")
            continue
        end
        # the shared-API table is Markdown, not a julia block; ignore empty noise
        try
            include_string(Main, code)
        catch e
            push!(failed, "block $i: $(sprint(showerror, e))")
        end
    end

    @test isempty(failed)
    if !isempty(failed)
        println("\nREADME example failures:")
        foreach(f -> println("  ✗ ", f), failed)
    end
    @test length(skipped) >= 1  # the GPU block is expected to be skipped
end