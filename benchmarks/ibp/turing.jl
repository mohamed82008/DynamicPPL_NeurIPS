using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using Turing
using Turing.RandomMeasures
using Statistics
using DataFrames
using LinearAlgebra
using Random: seed!
seed!(1)

@model function ibp(y, α, ::Type{M}=Vector{Int}) where {M}

    N = length(y)

    k = tzeros(Int, N)
    z = tzeros(Int, N, k)

    k[1] ~ Poisson(α)
    z[1,:] .= 1

    for i in 2:N
        K = size(z, 2)
        for j in 1:K
            mk = sum(z[:,j])
            z[i,k] ~ Bernoulli(mk / i)
        end
        k[i] ~ Poisson(α / i)
        if k[i] > 0
            z = hcat(z, tzeros(Int, N, k[i]))
            z[i,(K+1):end] .= 1
        end
    end

    K = size(z, 2)
    μ = M(undef, K)
    for k = 1:K
        μ[k] ~ Normal(0.0, 10.0)
    end

    for i in 1:N
        y[i] ~ Normal(μ' * z[i,:], 1.0)
    end 
end

x = [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78];
model = ibp(x, 10.0)

n_runs = 100
n_samples = 10_000
n_particles = 3

alg = Gibbs(HMC(0.01, 5, :μ), PG(n_particles, :z, :k))
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
        wandb.config.update(Dict("ppl" => "turing", "model" => "ibp"))
    end
    for runs in type_specialization
        times = []
        for i in 1:n_runs+1
            with_logger(NullLogger()) do    # disable numerical error warnings
                t = @elapsed sample(model, alg, n_samples; progress=false, chain_type=Any, specialize_after = runs)
                clog && i > 1 && wandb.log(Dict("time" => t))
                push!(times, t)
                push!(result, ("time_$runs", t, i, "ibp", "turing"))
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