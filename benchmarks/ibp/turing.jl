using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using Turing
using Turing.RandomMeasures
using Statistics
using DataFrames
using LinearAlgebra
using Random: seed!
seed!(1)

@model function ibp(y, α, kmax, ::Type{MV}=Vector{Float64}) where {MV}
   N = length(y)
   
   ks = tzeros(Int, N)
   ks[1] ~ Poisson(α)
   ks[1] = ks[1] <= kmax ? ks[1] : kmax
   
   z = tzeros(Int, N, kmax)
   z[1,1:ks[1]] .= 1

   for i in 2:N
       K = sum(ks[1:i-1])
       for j in 1:K
           mk = sum(z[:,j])
           z[i,j] ~ Bernoulli(mk / i)
       end
       ks[i] ~ Poisson(α / i)
       ks[i] = K+ks[i] <= kmax ? ks[i] : 0
       if ks[i] > 0
           z[i,(K+1):sum(ks[1:i])] .= 1
       end
   end

   K = sum(ks)
   μ = MV(undef, K)
   for j = 1:K
       μ[j] ~ Normal(0.0, 10.0)
   end

   for i in 1:N
       y[i] ~ Normal(μ' * z[i,1:K], 1.0)
   end
end

x = [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78];
model = ibp(x, 10.0, 100)

n_runs = 10
n_samples = 10_000
n_particles = 3

alg = Gibbs(HMC(0.01, 5, :μ), PG(n_particles, :z, :k))
chain = nothing

using BenchmarkTools
using Logging: with_logger, NullLogger

type_specialization = [0, 1, 10, 100]
result = DataFrame(type=[], value=[], mode=[], model=[], ppl=[])

if "--benchmark" in ARGS
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


