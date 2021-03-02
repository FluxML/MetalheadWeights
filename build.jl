using Pkg.Artifacts
using Pkg.TOML

WEIGHT_DIR = joinpath(@__DIR__, "weights")
BUILD_DIR = joinpath(@__DIR__, "build")
URL = "https://github.com/darsnack/MetalheadWeights/releases/download"
ARTIFACT_TOML = joinpath(@__DIR__, "Artifacts.toml")

function build(version; weights = WEIGHT_DIR,
                        build = BUILD_DIR,
                        artifact_toml = ARTIFACT_TOML,
                        url = URL)
    # Clean up prior builds.
    ispath(build) && rm(build; recursive=true, force=true)
    mkdir(build)

    # Package up weights
    for weight in readdir(weights)
        model = basename(weight)
        artifact_filename = "$model-$version.tar.gz"
        product_hash = create_artifact() do artifact_dir
            mkdir(joinpath(artifact_dir, model))
            cp(joinpath(weights, weight), joinpath(artifact_dir, model); force=true)
        end
        download_hash = archive_artifact(product_hash, joinpath(build, artifact_filename))
        remote_url = "$url/v$version/$artifact_filename"
        
        @info "Creating:" model version product_hash artifact_filename download_hash remote_url
        bind_artifact!(
            artifact_toml,
            "$model",
            product_hash,
            force=true,
            lazy=true,
            download_info=Tuple[(remote_url, download_hash)])
    end
end