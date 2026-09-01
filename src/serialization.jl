# ──────────────────────────────────────────────────────────────────────────────
# Serialization — save and load Canapes models
# ──────────────────────────────────────────────────────────────────────────────

const CANAPES_SERIALIZATION_VERSION = 2

"""
    save_model(model, path::String)

Serialize a Canapes model to disk using Julia's native serialization.
Includes a version header and type information for forward-compatibility checking.

The model is first written to a temporary file in the target directory and then
atomically renamed into place, so a crash or failed write never leaves a
partially-written model at `path` (atomic on POSIX; replace semantics on
Windows).

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 50, 20, 0.1);

julia> model = EASE(λ=100.0, verbose=false);

julia> fit!(model, X);

julia> path = tempname() * ".jls";

julia> save_model(model, path);

julia> size(load_model(path).B)
(20, 20)

julia> rm(path; force=true);
```
"""
function save_model(model::AbstractSparseModel, path::String)
    isdir(path) && throw(ArgumentError("save path is a directory: $path"))
    dir = dirname(path)
    !isempty(dir) && mkpath(dir)
    # Same directory as the target: the rename is atomic on the filesystem.
    tmp_path, io = mktemp(isempty(dir) ? "." : dir)
    try
        # Version header (v2 includes package version)
        write(io, "CANAPES_v$(CANAPES_SERIALIZATION_VERSION)\n")
        serialize(io, string(typeof(model)))
        serialize(io, model)
        flush(io)
        close(io)
        mv(tmp_path, path; force=true)
    catch
        close(io)
        rm(tmp_path; force=true)
        rethrow()
    end
    nothing
end

"""
    load_model(path::String) -> AbstractSparseModel

Deserialize a model from disk. Verifies the version header.

!!! warning "Security"
    Only load models from trusted sources. Julia's `Serialization` module
    can execute arbitrary code during deserialization.

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 50, 20, 0.1);

julia> model = EASE(λ=100.0, verbose=false);

julia> fit!(model, X);

julia> path = tempname() * ".jls";

julia> save_model(model, path);

julia> loaded = load_model(path);

julia> size(recommend(loaded, X; k=3))
(50, 3)

julia> rm(path; force=true);
```
"""
function load_model(path::String)
    isfile(path) || error("Model file not found: $path")
    open(path, "r") do io
        header = readline(io)
        startswith(header, "CANAPES_v") ||
            error("Invalid model file: missing CANAPES header in '$path'")
        # Parse version
        version_str = replace(header, "CANAPES_v" => "")
        version = tryparse(Int, version_str)
        version === nothing && throw(ArgumentError("Invalid version in header: '$header'"))
        version == CANAPES_SERIALIZATION_VERSION || throw(ArgumentError(
            "Unsupported Canapes model version $version; expected $CANAPES_SERIALIZATION_VERSION"))
        serialized_type = deserialize(io)
        serialized_type isa String || throw(ArgumentError(
            "Invalid model type metadata in '$path'"))
        model = deserialize(io)
        model isa AbstractSparseModel || throw(ArgumentError(
            "Serialized object is not a Canapes model: $serialized_type"))
        model
    end
end
