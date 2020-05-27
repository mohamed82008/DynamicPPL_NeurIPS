using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using Turing
using Turing.RandomMeasures
using LinearAlgebra
using Random: seed!
seed!(1)

@model imm(y, alpha, ::Type{M}=Vector{Float64}) where {M} = begin
    N = length(y)
    rpm = DirichletProcess(alpha)
    z = tzeros(Int, N)
    cluster_counts = tzeros(Int, N)
    for i in 1:N
        z[i] ~ ChineseRestaurantProcess(rpm, cluster_counts)
        cluster_counts[z[i]] += 1
    end
    Kmax = findlast(!iszero, cluster_counts)
    m = M(undef, Kmax)
    for k = 1:Kmax
        m[k] ~ Normal(1.0, 1.0)
    end
    y ~ TuringMvNormal(m[z], 1.0)
end

# sample data from prior
y = vcat(randn(10), randn(10).+2, randn(10).-2)
model = imm(y, 1.0)

include("../infer_turing_dynamic.jl")

;
