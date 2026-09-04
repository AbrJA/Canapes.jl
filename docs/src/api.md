# API Reference

## Core Types & Utilities

```@autodocs
Modules = [Canapes]
Pages = [
	"types.jl",
	"utils.jl",
	"sparse_utils.jl",
]
```

## Namespaced singletons

```@docs
Links
Sampling
```

## Experimental

```@docs
Canapes.Experimental
Canapes.Experimental.ProbabilisticMF
```

## Callbacks

```@autodocs
Modules = [Canapes]
Pages = ["callbacks.jl"]
```

## Cross-Validation & Hyperparameter Search

```@autodocs
Modules = [Canapes]
Pages = ["crossval.jl"]
```

## Serialization

```@autodocs
Modules = [Canapes]
Pages = ["serialization.jl"]
```

## SoftImpute family

```@docs
AbstractSoftALS
```

## Tables.jl Integration

```@autodocs
Modules = [Canapes]
Pages = ["tables.jl"]
```
