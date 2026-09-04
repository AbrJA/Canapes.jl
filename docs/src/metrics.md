# Ranking Metrics

```@docs
ap_at_k
mean_ap_at_k
ndcg_at_k
precision_at_k
recall_at_k
```

## Error Metrics (explicit ratings)

```@docs
rmse
mae
mean_rmse
mean_mae
```

## Example

```julia
using Canapes, SparseArrays

# Ground truth: user 1 likes items 3, 7, 9
actual = sparse([1,1,1], [3,7,9], ones(3), 1, 10)

# Model predictions: top-4 items for user 1
predictions = [3 7 1 9]

mean_ap_at_k(predictions, actual; k=4)     # scalar (macro-averaged) ≈ 0.833
ndcg_at_k(predictions, actual; k=4)        # per-user vector → [1.0]; scalar: mean(ndcg_at_k(...))
precision_at_k(predictions, actual; k=4)   # per-user vector → [0.75]
recall_at_k(predictions, actual; k=4)      # per-user vector → [1.0]
```

## Typical Workflow

```julia
using Canapes, SparseArrays, Random

X = sprand(MersenneTwister(42), 1000, 500, 0.02)
X_train, X_test = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(1))

model = ShallowAutoencoder(λ=500.0, verbose=false)
fit!(model, X_train)
preds = recommend(model, X_train; k=10)

# Evaluate all metrics at once (ndcg/precision/recall are per-user vectors → mean)
println("MAP@10:       ", round(mean_ap_at_k(preds, X_test; k=10), digits=4))
println("Mean NDCG@10: ", round(mean(ndcg_at_k(preds, X_test; k=10)), digits=4))
println("Precision@10: ", round(mean(precision_at_k(preds, X_test; k=10)), digits=4))
println("Recall@10:    ", round(mean(recall_at_k(preds, X_test; k=10)), digits=4))
```
