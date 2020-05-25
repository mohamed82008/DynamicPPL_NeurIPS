using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Memoization, Turing

@model high_dim_gauss(D) = begin
    m ~ filldist(Normal(0, 1), D)
end

model = high_dim_gauss(data["D"])

step_size = 0.1
n_steps = 4
test_zygote = false
test_tracker = true

include("../infer_turing.jl")

;
