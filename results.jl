using DataFrames, Plots, StatsPlots, Latexify, CSV, LaTeXStrings
using ArgParse, Printf, Statistics
pgfplotsx()

s = ArgParseSettings()

@add_arg_table! s begin
"--plots"
    action = :store_true
"--tables"
    action = :store_true
end

args = parse_args(s)

resultsdir = joinpath(pwd(), "benchmarks", "results")

resultsfiles = filter(fname -> startswith(fname, "result-") && endswith(fname, ".csv") && !endswith(fname, "stan-spec0.csv"), readdir(resultsdir))

typed = mapreduce(f -> begin
                    @info f
                    CSV.read(joinpath(resultsdir, f))
                    end, vcat, filter(fname -> endswith(fname, "spec1.csv"), resultsfiles));

untyped = mapreduce(f -> CSV.read(joinpath(resultsdir, f)), vcat, filter(fname -> endswith(fname, "spec0.csv"), resultsfiles));
untyped.ppl .= "turing(untyped)"

data = vcat(typed, untyped);

models = unique(data.model)
ppls = unique(data.ppl)

if args["plots"]
    # make plots folder
    mkpath(joinpath(pwd(), "plots"))

    for model in models
        p = @df data[(data.type .== "time") .& (data.model .== model),:] boxplot(:ppl, :value, ylabel = "runtime (seconds)", legend=nothing)
        savefig(p, joinpath(pwd(), "plots", string(model, "-boxplot.pdf")))
    end
end

if args["tables"]
    # create latex table
    rtable = DataFrame(Array{Any}(undef,0,length(ppls)+1), vcat("", map(m -> LaTeXString(m), ppls)))

    for model in models
        pplrow = map(ppl -> begin
            drow = data[(data.type .== "time") .& (data.model .== model) .& (data.ppl .== ppl), :value]
            μ = @sprintf("%.3f",mean(drow))
            σ = @sprintf("%.3f",std(drow))
            s = string(μ, " \\pm ", σ)
            LaTeXString(s)
            end, ppls)
        push!(rtable, vcat(LaTeXString(model), pplrow))
    end

    ltabel = latexify(rtable, env=:table, booktabs=true)
    open("results_table.tex", "w") do io
       write(io, ltabel)
    end
    
    # create latex table
    rtable = DataFrame(Array{Any}(undef,0,length(ppls)+1), vcat("", map(m -> LaTeXString(m), ppls)))

    for model in models
        pplrow = map(ppl -> begin
            drow = data[(data.type .== "time_forward") .& (data.model .== model) .& (data.ppl .== ppl), :value]
            μ = @sprintf("%.3E",mean(drow))
            s = string(μ)
            LaTeXString(s)
            end, ppls)
        push!(rtable, vcat(LaTeXString(model), pplrow))
    end

    ltabel = latexify(rtable, env=:table, booktabs=true)
    open("forward_results_table.tex", "w") do io
       write(io, ltabel)
    end
end
