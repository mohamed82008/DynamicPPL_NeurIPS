using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Memoization, Turing
using DistributionsAD: TuringMvNormal

@model naive_bayes(image_vec, label, D, N, C) = begin
    m ~ filldist(Normal(0, 10), C, D)
    image_vec ~ TuringMvNormal(vec(m[label,:]), 1.0)
end

N = 10
D = data["D"]
model = naive_bayes(vec(data["image"][1:N, 1:D]), data["label"][1:N], data["D"], data["N"], data["C"])

step_size = 0.1
n_steps = 4

test_zygote = false
test_tracker = false

include("../infer_turing.jl")

# Before uncommenting the code below, make sure chain_type is not Any in infer_turing.jl
#=
# Save result

if !isnothing(chain)
    using BSON

    m_data = chain[:m].value.data

    m_bayes = mean(
        map(
            i -> reconstruct(pca, Matrix{Float64}(reshape(m_data[i,:,1], D_pca, 10))),
            1_000:100:2_000
        )
    )

    bson("result.bson", m_bayes=m_bayes)
end
=#
