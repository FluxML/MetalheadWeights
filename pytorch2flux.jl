# Converts the weigths of a PyTorch model to a Flux model from Metalhead

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


function _list_state(node::Flux.BatchNorm,channel,prefix)
    # use the same order of parameters than PyTorch
    put!(channel, (prefix * ".γ", node.γ)) # weigth (learnable)
    put!(channel, (prefix * ".β", node.β)) # bias (learnable)
    put!(channel, (prefix * ".μ", node.μ))  # running mean
    put!(channel, (prefix * ".σ²", node.σ²)) # running variance
end

function _list_state(node::Union{Flux.Conv,Flux.Dense},channel,prefix)
    put!(channel, (prefix * ".weight", node.weight))

    if node.bias !== Flux.Zeros()
        put!(channel, (prefix * ".bias", node.bias))
    end
end

_list_state(node,channel,prefix) = nothing

function _list_state(node::Union{Flux.Chain,Flux.Parallel},channel,prefix)
    for (i,n) in enumerate(node.layers)
        _list_state(n,channel,prefix * ".layers[$i]")
    end
end

function list_state(node; prefix = "model")
    Channel() do channel
        _list_state(node,channel,prefix)
    end
end

for (modelname,jlmodel,pymodel) in modellib

    model = jlmodel()
    pytorchmodel = pymodel(pretrained=true)

    state = OrderedDict(list_state(model.layers))

    # pytorchmodel.state_dict() looses the order
    state_dict = OrderedDict(pycall(pytorchmodel.state_dict,PyObject).items())
    pytorch_pp = OrderedDict((k,v.numpy()) for (k,v) in state_dict if !occursin("num_batches_tracked",k))


    # loop over all parameters
    for ((flux_key,flux_param),(pytorch_key,pytorch_param)) in zip(state,pytorch_pp)
        if size(flux_param) == size(pytorch_param)
            # Dense weight and vectors
            flux_param .= pytorch_param
        elseif size(flux_param) == reverse(size(pytorch_param))
            tmp = pytorch_param
            tmp = permutedims(tmp,ndims(tmp):-1:1)

            if ndims(flux_param)  == 4
                # convolutional weights
                flux_param .= reverse(tmp,dims=(1,2))
            else
                flux_param .= tmp
            end
        else
            @debug begin
                @show size(flux_param), size(pytorch_param)
            end
            error("incompatible shape $flux_key $pytorch_key")
        end
    end

    @info "saving model $modelname"
    BSON.@save joinpath(@__DIR__,"weights","$(modelname).bson") model
end
