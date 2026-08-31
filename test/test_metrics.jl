# test/test_metrics.jl — Ranking metrics tests

@testset "Perfect ranking" begin
    actual = sparse([1,1,1], [5,7,9], ones(3), 1, 10)
    predictions = [5 7 9 2]

    ap = ap_at_k(predictions, actual; k=4)
    @test length(ap) == 1
    @test ap[1] ≈ 1.0

    @test mean_ap_at_k(predictions, actual; k=4) ≈ 1.0
    @test ndcg_at_k(predictions, actual; k=4)[1] ≈ 1.0
    @test precision_at_k(predictions, actual; k=4)[1] ≈ 0.75
    @test recall_at_k(predictions, actual; k=4)[1] ≈ 1.0
end

@testset "No hits" begin
    actual = sparse([1,1], [8,9], ones(2), 1, 10)
    preds = [1 2 3 4]
    @test ap_at_k(preds, actual; k=4)[1] ≈ 0.0
    @test ndcg_at_k(preds, actual; k=4)[1] ≈ 0.0
    @test precision_at_k(preds, actual; k=4)[1] ≈ 0.0
end

@testset "Partial recall" begin
    actual = sparse([1,1,1,1,1], 1:5, ones(5), 1, 10)
    preds = [1 2]
    @test precision_at_k(preds, actual; k=2)[1] ≈ 1.0
    @test recall_at_k(preds, actual; k=2)[1] ≈ 2/5
end

@testset "AP ordering sensitivity" begin
    actual = sparse([1,1], [1,3], ones(2), 1, 5)
    p_first = [1 2 3]
    p_delayed = [2 1 3]
    @test ap_at_k(p_first, actual; k=3)[1] > ap_at_k(p_delayed, actual; k=3)[1]
end

@testset "Multiple users" begin
    actual_multi = sparse([1,1,2,2], [1,3,2,4], ones(4), 2, 5)
    preds_multi = [1 3; 2 4]
    ap_multi = ap_at_k(preds_multi, actual_multi; k=2)
    @test length(ap_multi) == 2
    @test all(ap_multi .≈ 1.0)
end

@testset "Multi-user batch: all-perfect" begin
    actual_m = sparse([1,1,2,2,3,3], [1,2,3,4,5,6], ones(6), 3, 10)
    preds_m = [1 2; 3 4; 5 6]
    @test all(ap_at_k(preds_m, actual_m; k=2) .≈ 1.0)
    @test all(ndcg_at_k(preds_m, actual_m; k=2) .≈ 1.0)
end

@testset "No relevant items" begin
    actual_empty = sparse(Int[], Int[], Float64[], 1, 10)
    preds_empty = [1 2 3]
    @test ap_at_k(preds_empty, actual_empty; k=3)[1] ≈ 0.0
    @test precision_at_k(preds_empty, actual_empty; k=3)[1] ≈ 0.0
    @test recall_at_k(preds_empty, actual_empty; k=3)[1] ≈ 0.0
end

@testset "NDCG with graded relevance" begin
    # Items with different relevance scores
    actual = sparse([1,1,1], [1,2,3], [3.0, 2.0, 1.0], 1, 5)
    # Best order: [1, 2, 3]
    preds_best = [1 2 3]
    preds_worst = [3 2 1]
    @test ndcg_at_k(preds_best, actual; k=3)[1] ≈ 1.0
    @test ndcg_at_k(preds_worst, actual; k=3)[1] < 1.0
end

@testset "k smaller than predictions width" begin
    actual = sparse([1,1,1], [1,2,3], ones(3), 1, 10)
    preds = [1 2 3 4 5]
    # k=2 should only consider first 2 predictions
    prec = precision_at_k(preds, actual; k=2)
    @test prec[1] ≈ 1.0  # both [1,2] are relevant
end

@testset "Invalid ranking inputs" begin
    actual = sparse([1], [1], [1.0], 1, 3)
    @test_throws ArgumentError precision_at_k([1 2], actual; k=0)
    @test_throws ArgumentError precision_at_k([1 1], actual; k=2)
    @test_throws ArgumentError precision_at_k([1 4], actual; k=2)
    @test_throws DimensionMismatch precision_at_k(reshape([1, 2], 2, 1), actual; k=1)
end
