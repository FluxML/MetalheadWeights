# MetalheadWeights

Pre-trained model weight artifacts for [Metalhead.jl](https://github.com/FluxML/Metalhead.jl).

## Building new weights

1. Add the BSON file (e.g. `vgg19.bson`) to the `weights/` directory. The filename will determine the name of the artifact.
2. Run `include("build.jl"); build(version)` where `version` is the version string (e.g. `build("0.1.0")`).
3. Upload the contents of `build/` to the releases page on Github (each `.tar.gz` is a separate artifact). Remember to set the version number correctly. For example, `build/vgg19-0.1.0.tar.gz` should be downloadable at `https://github.com/darsnack/MetalheadWeights/releases/download/v0.1.0/vgg19-0.1.0.tar.gz`.

## Using weight artifacts

Copy the contents of `Artifacts.toml` to your project. You can use each artifact using the `artifact"..."` syntax (e.g. `artifact"vgg19"`). The artifacts are lazy downloaded on first use.