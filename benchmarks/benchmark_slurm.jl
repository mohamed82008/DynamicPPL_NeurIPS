using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using ArgParse
using CSV

s = ArgParseSettings()

@add_arg_table! s begin
"model"
    arg_type = String
    required = true
"ppl"
    arg_type = String
    required = true
end

args = parse_args(s)

models = [
    "high_dim_gauss",
    "gauss_unknown",
    "h_poisson",
    "hmm_semisup",
    "naive_bayes",
    "logistic_reg",
    "sto_volatility",
    "lda",
    "lda_unvectorized",

    # dynamic models
    "stochastic_control_flow",
]

MODEL = args["model"]
@assert MODEL in models

ppls = [
    "turing", 
    "stan",
]

PPL = args["ppl"]
@assert PPL in ppls

push!(ARGS, "--benchmark")

if isfile(projectdir("benchmarks", MODEL, "$PPL.jl"))
	@info "Benchmarking $MODEL using $PPL ..."

        withenv("MODEL_NAME" => MODEL) do
		include(projectdir("benchmarks", MODEL, "$PPL.jl"))

		@assert @isdefined result
		CSV.write(projectdir("benchmarks", "results.csv"), result, append=true)
	end

else
	@warn "skipping $MODEL using $PPL ... "
end
