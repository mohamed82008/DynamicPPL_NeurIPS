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

script_exists = isfile(projectdir("benchmarks", MODEL, "$PPL.jl"))
result_exists = isfile(projectdir("benchmarks", "results", "result-$MODEL-$PPL-spec0.csv"))

if script_exists && !result_exists
    @info "Benchmarking $MODEL using $PPL ..."

    withenv("MODEL_NAME" => MODEL, "TYPING" => 0) do
        include(projectdir("benchmarks", MODEL, "$PPL.jl"))

        @assert @isdefined result
        CSV.write(projectdir("benchmarks", "results", "result-$MODEL-$PPL-spec0.csv"), result)
        #CSV.write(projectdir("benchmarks", "results.csv"), result, append=true)
    end
else
    @warn "skipping $MODEL using $PPL ... "
    @info "script exists: $script_exists, results exists: $result_exists"
end

script_exists = isfile(projectdir("benchmarks", MODEL, "$PPL.jl"))
result_exists = isfile(projectdir("benchmarks", "results", "result-$MODEL-$PPL-spec1.csv"))

if script_exists && !result_exists
    @info "Benchmarking $MODEL using $PPL ..."

    withenv("MODEL_NAME" => MODEL, "TYPING" => 1) do
        include(projectdir("benchmarks", MODEL, "$PPL.jl"))

        @assert @isdefined result
        CSV.write(projectdir("benchmarks", "results", "result-$MODEL-$PPL-spec1.csv"), result)
        #CSV.write(projectdir("benchmarks", "results.csv"), result, append=true)
    end
else
    @warn "skipping $MODEL using $PPL ... "
    @info "script exists: $script_exists, results exists: $result_exists"
end
