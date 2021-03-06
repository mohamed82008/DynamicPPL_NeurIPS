# alg = CmdStan.Hmc(
    # CmdStan.Static(n_steps * step_size),
    # CmdStan.diag_e(),
    # step_size,
    # 0.0,
# )

# model = Stanmodel(
    # model=model_str,
    # nchains=1,
    # Sample(
        # algorithm=alg,
        # num_warmup=0,
        # num_samples=10_000,
        # adapt=CmdStan.Adapt(engaged=false),
        # save_warmup=true,
    # ),
    # printsummary=false,
    # output_format=:array,
    # tmpdir=joinpath(pwd(), "tmpjl")
# )

n_runs = 10

using BenchmarkTools
using DataFrames

using PyCall: pyimport
pystan = pyimport("pystan")
sm = pystan.StanModel(model_code=model_str,
  extra_compile_args = ["-ftemplate-depth-256", "-O3",
   "-mtune=native", "-march=native", "-pipe", "-fno-trapping-math",
    "-funroll-loops", "-funswitch-loops"])
fit_stan(n_iters=10_000) = sm.sampling(
    data=data, iter=n_iters, chains=1, warmup=0, algorithm="HMC",
    control=Dict(
        "adapt_engaged" => false,
        # HMC
        "int_time" => n_steps * step_size,
        "metric" => "diag_e",
        "stepsize" => step_size,
        "stepsize_jitter" => 0,
    )
)
f = fit_stan(100)
theta = f.unconstrain_pars(f.get_last_position()[1])
forward_model(x) = f.log_prob(x)
gradient(x) = f.grad_log_prob(x)

result = DataFrame(type=[], value=[], mode=[], model=[], ppl=[])

if "--benchmark" in ARGS
    using Statistics: mean, std
    clog = "WANDB" in keys(ENV)    # cloud logging flag
    if clog
        # Setup W&B
        using PyCall: pyimport
        wandb = pyimport("wandb")
        wandb.init(project="turing-benchmark")
        wandb.config.update(Dict("ppl" => "stan", "model" => ENV["MODEL_NAME"]))
    end
    times = []
    for i in 1:n_runs
        t = @elapsed fit_stan()
        # Parse inference time from log
        # tl = read(pipeline(`tail tmp/noname_run.log`, `rg "s \(S"`), String)
        # t = parse(Float64, match(r"[0-9]+.[0-9]+", tl).match)
        push!(result, ("time", t, i, ENV["MODEL_NAME"], "stan"))
        push!(times, t)
        clog && wandb.log(Dict("time" => t))
    end
    t_mean = mean(times)
    t_std = std(times)
    t_forward = @belapsed $forward_model($theta)
    t_forward = @belapsed $forward_model($theta)
    t_gradient = @belapsed $gradient($theta)
    t_gradient = @belapsed $gradient($theta)
    
    push!(result, ("time_forward", t_forward, "", ENV["MODEL_NAME"], "stan"))
    push!(result, ("time_gradient", t_gradient, "", ENV["MODEL_NAME"], "stan"))
    
    println("Benchmark results")
    println("  Running time: $t_mean +/- $t_std ($n_runs runs)")
    println("  Forward time: $t_forward")
    println("  Gradient time: $t_gradient")
    if clog
        wandb.run.summary.time_mean     = t_mean
        wandb.run.summary.time_std      = t_std
        wandb.run.summary.time_forward  = t_forward
        wandb.run.summary.time_gradient = t_gradient
    end
elseif "--function" in ARGS
    @btime $forward_model($theta)
    @btime $gradient($theta)
else
    @time fit_stan()
end

result
