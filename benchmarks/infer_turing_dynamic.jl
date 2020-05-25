@assert @isdefined n_samples
@assert @isdefined n_particles

alg = PG(n_particles)

chain = nothing

using BenchmarkTools
using Logging: with_logger, NullLogger

type_specialization = [1, 2, 4, 8, 16]

if "--benchmark" in ARGS
    using Statistics: mean, std
    clog = "MODEL_NAME" in keys(ENV)    # cloud logging flag
    if clog
        # Setup W&B
        using PyCall: pyimport
        wandb = pyimport("wandb")
        wandb.init(project="turing-benchmark")
        wandb.config.update(Dict("ppl" => "turing", "model" => ENV["MODEL_NAME"]))
    end
    for runs in type_specialization
        n_runs = 3
        times = []
        for i in 1:n_runs+1
            with_logger(NullLogger()) do    # disable numerical error warnings
                t = @elapsed sample(model, alg, n_samples; progress=false, chain_type=Any, specialize_after = runs)
                clog && i > 1 && wandb.log(Dict("time" => t))
                push!(times, t)
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
