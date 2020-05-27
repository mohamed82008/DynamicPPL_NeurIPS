using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using Turing
using LinearAlgebra
using Random: seed!
seed!(1)

@model testmodel(p, O) = begin
    x ~ Categorical(p)
    if x == 1
        y ~ filldist(Normal(), length(O))
        O ~ TuringMvNormal(y, 1.0)
    elseif x == 2
        z ~ filldist(Normal(), length(O))
        O ~ TuringMvNormal(z, 1.0)
    else
        k ~ filldist(Normal(), length(O))
        O ~ TuringMvNormal(k, 1.0)
    end
end

# sample data from prior
N = 1000
p = [0.25, 0.5, 0.25]
O = Array{Float64}(testmodel(p, fill(missing, N))())

model = testmodel(p, O)

include("../infer_turing_dynamic.jl")

;
