using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using Turing
using Turing.RandomMeasures
using Statistics
using DataFrames
using LinearAlgebra
using Random: seed!
seed!(1)

@model function imm(y, α, ::Type{T}=Vector{Float64}) where T
    m = DirichletProcess(α)
    N = length(y)
    nk = tzeros(Int, N)
    z = tzeros(Int, N)

    for i in 1:N
        z[i] ~ ChineseRestaurantProcess(m,nk)
        nk[z[i]] += 1
    end

    K = findlast(!iszero, nk)

    μ = T(undef, K)
    s = T(undef, K)
    for k = 1:K
        μ[k] ~ Normal(0.0, 1.0)
        s[k] ~ InverseGamma(2.0, 3.0)
    end

    for i in 1:N
        x[i] ~ Normal(μ[z[i]], sqrt(s[z[i]]))
    end
end

# sample data from prior
x = [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78];
model = imm(x, 10.0)

n_runs = 100
n_samples = 10_000
n_particles = 3

alg = Gibbs(HMC(0.01, 5, :μ, :s), PG(n_particles, :z))
chain = nothing

using BenchmarkTools
using Logging: with_logger, NullLogger

type_specialization = [0, 1, 10, 100]
result = DataFrame(type=[], value=[], mode=[], model=[], ppl=[])

if "--benchmark" in ARGS
    using Statistics: mean, std
    
    for runs in type_specialization
        times = []
        for i in 1:n_runs+1
            with_logger(NullLogger()) do    # disable numerical error warnings
                seed!(i)
                generic_sampler = Turing.Sampler(alg, model, specialize_after=runs);
                empty!(generic_sampler);
                seed!(i)
                t = @elapsed sample(model, generic_sampler, n_samples; progress=false, chain_type=Any)
                push!(times, t)
            end
        end

        for i in 2:n_runs+1
            push!(result, ("time_$runs", t, i-1, "ibp", "turing"))
        end

        t_mean = mean(times[2:end])
        t_std = std(times[2:end])

        # Estimate compilation time
        t_with_compilation = times[1]
        t_compilation_approx = t_with_compilation - t_mean

        println("Benchmark results")
        println("  Compilation time ($runs): $t_compilation_approx (approximately)")
        println("  Running time ($runs): $t_mean +/- $t_std ($n_runs runs)")
    end
else
    @time chain = sample(model, alg, n_samples; progress_style=:plain, chain_type=Any)
end
result

;
