using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using Turing
using LinearAlgebra
using Random: seed!
seed!(1)

@model testmodel(p, O) = begin
    x ~ Categorical(p)
    if x == 1
        y ~ MvNormal(zeros(length(O)), 1.0)
        for i in 1:length(O)
            O[i] ~ Normal(y[i] * norm(y), 1.0)
        end
    elseif x == 2
        z ~ MvNormal(zeros(length(O)), 1.0)
        for i in 1:length(O)
            O[i] ~ Normal(z[i] * norm(z), 1.0)
        end
    else
        k ~ MvNormal(zeros(length(O)), 1.0)
        for i in 1:length(O)
            O[i] ~ Normal(k[i] * norm(k), 1.0)
        end
    end
    return O
end

# sample data from prior
N = 1000
p = [0.25, 0.5, 0.25]
O = Array{Float64}(testmodel(p, fill(missing, N))())

model = testmodel(p, O)

# inference
n_particles = 10
n_samples = 10_0

include("../infer_turing_dynamic.jl")

;
