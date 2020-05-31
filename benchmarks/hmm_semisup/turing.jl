using DrWatson
@quickactivate "DynamicPPL_NeurIPS"

using Random: seed!
seed!(1234)

include("data.jl")

data = get_data()

using Memoization, Turing
using DistributionsAD: MvCategorical
using StatsFuns: logsumexp

@model hmm_semisup(K, T_unsup, w, z, u, alpha, beta) = begin
    theta ~ filldist(Dirichlet(alpha), K)
    phi ~ filldist(Dirichlet(beta), K)
    w ~ MvCategorical(phi[:, z])
    z[2:end] ~ MvCategorical(theta[:, z[1:end-1]])

    TF = eltype(theta)
    acc = similar(alpha, TF, K)
    gamma = similar(alpha, TF, K)
    temp_gamma = similar(alpha, TF, K)
    for k in 1:K
        gamma[k] = log(phi[u[1],k])
    end

    for t in 2:T_unsup
        for k in 1:K
            logphi = log(phi[u[t],k])
            for j in 1:K
                acc[j] = gamma[j] + log(theta[k,j]) + logphi
            end
            temp_gamma[k] = logsumexp(acc)
        end
        gamma .= temp_gamma
    end
    Turing.acclogp!(_varinfo, logsumexp(gamma))
end

model = hmm_semisup(data["K"], data["T_unsup"], data["w"], data["z"], data["u"], data["alpha"], data["beta"])

step_size = 0.001
n_steps = 4
test_zygote = false
test_tracker = false

seed!(1)

#include("../infer_turing.jl")

using DataFrames
using ReverseDiff, Memoization, Zygote

alg = HMC(step_size, n_steps)
n_samples = 10_000
n_runs = 10

chain = nothing

using BenchmarkTools
using Logging: with_logger, NullLogger

if test_tracker && test_zygote
    ADBACKENDS = Dict(
        "forwarddiff" => Turing.Core.ForwardDiffAD{40},
        "reversediff" => Turing.Core.ReverseDiffAD{true},
        "tracker" => Turing.Core.TrackerAD,
        "zygote" => Turing.Core.ZygoteAD,
    )
elseif test_tracker
    ADBACKENDS = Dict(
        "forwarddiff" => Turing.Core.ForwardDiffAD{40},
        "reversediff" => Turing.Core.ReverseDiffAD{true},
        "tracker" => Turing.Core.TrackerAD,
    )
elseif test_zygote
    ADBACKENDS = Dict(
        "forwarddiff" => Turing.Core.ForwardDiffAD{40},
        "reversediff" => Turing.Core.ReverseDiffAD{true},
        "zygote" => Turing.Core.ZygoteAD,
    )
else
    ADBACKENDS = Dict(
        "forwarddiff" => Turing.Core.ForwardDiffAD{40},
        "reversediff" => Turing.Core.ReverseDiffAD{true},
    )
end

function get_eval_functions(step_size, n_steps, model)
    spl_prior = Turing.SampleFromPrior()
    function forward_model(x)
        vi = Turing.VarInfo(model)
        Turing.link(vi)[spl_prior] = x
        model(Turing.link(vi), spl_prior)
        Turing.getlogp(vi)
    end
    grad_funcs = map(values(ADBACKENDS)) do adbackend
        alg_ad = HMC{adbackend}(step_size, n_steps)
        vi = Turing.VarInfo(model)
        spl = Turing.Sampler(alg_ad, model)
        Turing.Core.link!(vi, spl, model)
        x -> Turing.Core.gradient_logp(adbackend(), x, Turing.link(vi), model, spl)
    end
    x = Turing.link(Turing.VarInfo(model))[Turing.SampleFromPrior()]
    return x, forward_model, grad_funcs
end

theta, forward_model, grad_funcs = get_eval_functions(step_size, n_steps, model)

result = DataFrame(type=[], value=[], mode=[], model=[], ppl=[])

if "--benchmark" in ARGS
    using Statistics: mean, std
    clog = "WANDB" in keys(ENV)    # cloud logging flag
    if clog
        # Setup W&B
        using PyCall: pyimport
        wandb = pyimport("wandb")
        wandb.init(project="turing-benchmark")
        wandb.config.update(Dict("ppl" => "turing", "model" => ENV["MODEL_NAME"]))
    end
    times = []

    seed!(1)
    varinfo = Turing.VarInfo(model);
    model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext());
    init_theta = varinfo[Turing.SampleFromPrior()]

    for i in 1:n_runs+1
        with_logger(NullLogger()) do    # disable numerical error warnings
            t = @elapsed sample(model, alg, n_samples; progress=false, chain_type=Any, init_theta = init_theta)
            clog && i > 1 && wandb.log(Dict("time" => t))
            push!(times, t)
        end
    end
    t_mean = mean(times[2:end])
    t_std = std(times[2:end])
    # Estimate compilation time
    t_with_compilation = times[1]
    t_compilation_approx = t_with_compilation - t_mean

    t_forward = @belapsed $forward_model($theta)
    t_forward = @belapsed $forward_model($theta)

    push!(result, ("time_compilation", t_compilation_approx, "", ENV["MODEL_NAME"], "turing"))
    for i in 2:n_runs+1
        push!(result, ("time", times[i], i-1, ENV["MODEL_NAME"], "turing"))
    end
    push!(result, ("time_forward", t_forward, "", ENV["MODEL_NAME"], "turing"))

    println("Benchmark results")
    println("  Compilation time: $t_compilation_approx (approximately)")
    println("  Running time: $t_mean +/- $t_std ($n_runs runs)")
    println("  Forward time: $t_forward")

    if clog
        wandb.run.summary.time_mean = t_mean
        wandb.run.summary.time_std = t_std
        wandb.run.summary.time_forward = t_forward
    end
    for (name, grad_func) in zip(keys(ADBACKENDS), grad_funcs)
        t = @belapsed $grad_func($theta)
        t = @belapsed $grad_func($theta)
        push!(result, ("time_gradient", t, name, ENV["MODEL_NAME"], "turing"))
        println("  Gradient time ($name): $t")
        if clog
            s = Symbol("time_gradient_$name")
            eval(:(wandb.run.summary.$s=$t))
        end
    end
elseif "--function" in ARGS
    println("Forward time")
    @btime $forward_model($theta)
    for (name, grad_func) in zip(keys(ADBACKENDS), grad_funcs)
        println("Gradient time ($name)")
        @btime $grad_func($theta)
    end
else
    @time chain = sample(model, alg, n_samples; progress_style=:plain, chain_type=Any)
end

result
