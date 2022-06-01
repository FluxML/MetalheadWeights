# Compare Flux model from Metalhead to PyTorch model
# for a sample image

# PyTorch need to be installed

# Tested on ResNet and VGG models

using Flux
import Metalhead
using DataStructures
using Statistics
using BSON
using PyCall
using Images
using Test

using MLUtils
using Random

torchvision = pyimport("torchvision")
torch = pyimport("torch")

modellib = [
    ("vgg11", () -> Metalhead.VGG(11), torchvision.models.vgg11),
    ("vgg13", () -> Metalhead.VGG(13), torchvision.models.vgg13),
    ("vgg16", () -> Metalhead.VGG(16), torchvision.models.vgg16),
    ("vgg19", () -> Metalhead.VGG(19), torchvision.models.vgg19),
    ("resnet18", () -> Metalhead.ResNet(18), torchvision.models.resnet18),
    ("resnet34", () -> Metalhead.ResNet(34), torchvision.models.resnet34),
    ("resnet50", () -> Metalhead.ResNet(50), torchvision.models.resnet50),
    ("resnet101",() -> Metalhead.ResNet(101),torchvision.models.resnet101),
    ("resnet152",() -> Metalhead.ResNet(152),torchvision.models.resnet152),
]


tr(tmp) = permutedims(tmp,ndims(tmp):-1:1)


function normalize(data)
    cmean = reshape(Float32[0.485, 0.456, 0.406],(1,1,3,1))
    cstd = reshape(Float32[0.229, 0.224, 0.225],(1,1,3,1))
    return (data .- cmean) ./ cstd
end

# test image
guitar_path = download("https://cdn.pixabay.com/photo/2015/05/07/11/02/guitar-756326_960_720.jpg")

# image net labels
labels = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))

weightsdir = joinpath(@__DIR__,"..","weights")

for (modelname,jlmodel,pymodel) in modellib
    println(modelname)

    model = jlmodel()

    saved_model = BSON.load(joinpath(weightsdir,"$(modelname).bson"))
    Flux.loadmodel!(model,saved_model[:model])

    pytorchmodel = pymodel(pretrained=true)

    Flux.testmode!(model)

    sz = (224, 224)
    img = Images.load(guitar_path);
    img = imresize(img, sz);
    # CHW -> WHC
    data = permutedims(convert(Array{Float32}, channelview(img)), (3,2,1))
    data = normalize(data[:,:,:,1:1])

    out = model(data) |> softmax;
    out = out[:,1]

    println("  Flux:")

    for i in sortperm(out,rev=true)[1:5]
        println("    $(labels[i]): $(out[i])")
    end


    pytorchmodel.eval()
    output =  pytorchmodel(torch.Tensor(tr(data)));
    probabilities = torch.nn.functional.softmax(output[0], dim=0).detach().numpy();

    println("  PyTorch:")

    for i in sortperm(probabilities[:,1],rev=true)[1:5]
        println("    $(labels[i]): $(probabilities[i])")
    end

    @test maximum(out) ≈ maximum(probabilities)
    @test argmax(out) ≈ argmax(probabilities)

    println()
end
