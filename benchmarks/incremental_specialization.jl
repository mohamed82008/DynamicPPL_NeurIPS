using Random, Turing, DynamicPPL
using DistributionsAD: TuringMvNormal

Turing.setadbackend(:forwarddiff)

@model demo(p, F, O) = begin
    x ~ Categorical(p)
    if x == 1
        y ~ filldist(Normal(), size(F, 2))
        O ~ TuringMvNormal(F * y, 1.0)
    elseif x == 2
        z ~ filldist(Normal(), size(F, 2))
        O ~ TuringMvNormal(F * z, 1.0)
    else
        k ~ filldist(Normal(), size(F, 2))
        O ~ TuringMvNormal(F * k, 1.0)
    end
end

Random.seed!(1)
model = demo(fill(1/3, 3), rand(1000, 10), rand(1000))
alg = Gibbs(PG(20, :x), HMC(0.1, 4, :y, :z, :k))

# No type specialization

Random.seed!(1)
generic_sampler = Turing.Sampler(alg, model, specialize_after=0);
empty!(generic_sampler);
Random.seed!(1)
chain = sample(model, generic_sampler, 10000); # 8 min 32 sec

# Specialized for 2 RVs: x and y

Random.seed!(1)
specialized_sampler_1 = Turing.Sampler(alg, model, specialize_after=1);
empty!(specialized_sampler_1);
@show DynamicPPL.getinferred(specialized_sampler_1);
Random.seed!(1)
chain = sample(model, specialized_sampler_1, 10000); # 6 min 48 sec

# Specialized for 3 RVs: x, y and k

Random.seed!(1)
specialized_sampler_2 = Turing.Sampler(alg, model, specialize_after=2);
empty!(specialized_sampler_2);
@show DynamicPPL.getinferred(specialized_sampler_2);
Random.seed!(1)
chain = sample(model, specialized_sampler_2, 10000); # 2 min 14 sec

# Specialized for 4 RVs: x, y, k and z

Random.seed!(1)
specialized_sampler_3 = Turing.Sampler(alg, model, specialize_after=4);
empty!(specialized_sampler_3);
@show DynamicPPL.getinferred(specialized_sampler_3);
Random.seed!(1)
chain = sample(model, specialized_sampler_3, 10000); # 0 min 51 sec
