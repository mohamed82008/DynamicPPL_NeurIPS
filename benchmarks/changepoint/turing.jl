using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using Turing
using Turing.RandomMeasures
using Statistics
using DataFrames
using LinearAlgebra
using Random: seed!
seed!(1)

@model function changepoint(y)
    α = 1/mean(y)
    λ₁ = Exponential(α)
    λ₂ = Exponential(α)
    τ = DiscreteUniform(1, length(y))
    for i in eachindex(y)
        z[i] ~ Poisson(τ > i ? λ₁ : λ₂)
    end
    return τ
end

# sample data from prior
x = vcat(rand(Poisson(4), 50), rand(Poisson(10), 50))
model = changepoint(x)

n_runs = 100
n_samples = 10_000
n_particles = 3

alg = Gibbs(HMC(0.01, 5, :λ₁, :λ₂), PG(n_particles, :τ))
chain = nothing

using BenchmarkTools
using Logging: with_logger, NullLogger

type_specialization = [0, 1, 10, 100]
result = DataFrame(type=[], value=[], mode=[], model=[], ppl=[])

if "--benchmark" in ARGS
    using Statistics: mean, std
    clog = "WANDB" in keys(ENV)    # cloud logging flag
    if clog
        # Setup W&B
        using PyCall: pyimport
        wandb = pyimport("wandb")
        wandb.init(project="turing-benchmark")
        wandb.config.update(Dict("ppl" => "turing", "model" => "changepoint"))
    end
    for runs in type_specialization
        times = []
        for i in 1:n_runs+1
            with_logger(NullLogger()) do    # disable numerical error warnings
                t = @elapsed sample(model, alg, n_samples; progress=false, chain_type=Any, specialize_after = runs)
                clog && i > 1 && wandb.log(Dict("time" => t))
                push!(times, t)
                push!(result, ("time_$runs", t, i, "changepoint", "turing"))
            end
        end

        t_mean = mean(times[2:end])
        t_std = std(times[2:end])

        # Estimate compilation time
        t_with_compilation = times[1]
        t_compilation_approx = t_with_compilation - t_mean

        println("Benchmark results")
        println("  Compilation time ($runs): $t_compilation_approx (approximately)")
        println("  Running time ($runs): $t_mean +/- $t_std ($n_runs runs)")
        if clog
            s = Symbol("time_mean_$runs")
            eval(:(wandb.run.summary.$s=$(t_mean)))
            s = Symbol("time_std_$runs")
            eval(:(wandb.run.summary.$s=$(t_std)))
        end
    end
else
    @time chain = sample(model, alg, n_samples; progress_style=:plain, chain_type=Any)
end
result

;
