using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using ArgParse
using CSV
using LinearAlgebra

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
    "ibp",
    "crp",
    "changepoint",
    "demo_cflow",
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

specs = PPL == "stan" ? [1] : [1,0]

for spec in specs
    script_exists = isfile(projectdir("benchmarks", MODEL, "$PPL.jl"))
    result_exists = isfile(projectdir("benchmarks", "results", "result-$MODEL-$PPL-spec$spec.csv"))

    if script_exists && !result_exists
        @info "Benchmarking $MODEL using $PPL - Spec: $spec ..."
        @info "BLAS: $(BLAS.vendor())"

        withenv("MODEL_NAME" => MODEL, "TYPING" => spec) do
            include(projectdir("benchmarks", MODEL, "$PPL.jl"))

            @assert @isdefined result
            CSV.write(projectdir("benchmarks", "results", "result-$MODEL-$PPL-spec$spec.csv"), result)
            #CSV.write(projectdir("benchmarks", "results.csv"), result, append=true)
        end
    else
        @warn "skipping $MODEL using $PPL ... "
        @info "script exists: $script_exists, results exists: $result_exists"
    end
end
