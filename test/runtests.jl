using Test
using Gideon
using SparseArrays
using LinearAlgebra
using Random
using Statistics
using Aqua
using JET
using Pkg

@testset verbose=true "Gideon.jl" begin
    @testset "Quality" begin
        include("test_quality.jl")
    end
    @testset "Types & Utils" begin
        include("test_utils.jl")
    end
    @testset "WMF" begin
        include("test_wrmf.jl")
    end
    @testset "IALS" begin
        include("test_ials.jl")
    end
    @testset "EALS" begin
        include("test_eals.jl")
    end
    @testset "FTRL" begin
        include("test_ftrl.jl")
    end
    @testset "FM" begin
        include("test_fm.jl")
    end
    @testset "GloVe" begin
        include("test_glove.jl")
    end
    @testset "LogisticMF" begin
        include("test_lmf.jl")
    end
    @testset "BPR" begin
        include("test_bpr.jl")
    end
    @testset "EASE" begin
        include("test_ease.jl")
    end
    @testset "SLIM" begin
        include("test_slim.jl")
    end
    @testset "ADMMSLIM" begin
        include("test_admmslim.jl")
    end
    @testset "ItemKNN" begin
        include("test_knn.jl")
    end
    @testset "SoftImpute" begin
        include("test_soft_impute.jl")
    end
    @testset "Metrics" begin
        include("test_metrics.jl")
    end
    @testset "Infrastructure" begin
        include("test_infrastructure.jl")
    end
    @testset "Tables" begin
        include("test_tables.jl")
    end
    @testset "GPU" begin
        include("test_gpu.jl")
    end
    @testset "Coverage" begin
        include("test_coverage.jl")
    end
    @testset "Correctness" begin
        include("test_correctness.jl")
    end
    @testset "Reference contracts" begin
        include("test_reference_contracts.jl")
    end
end
