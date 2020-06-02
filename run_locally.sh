#!/bin/bash

# static models
for dataset in high_dim_gauss gauss_unknown h_poisson hmm_semisup naive_bayes logistic_reg sto_volatility lda lda_unvectorized; do
    for ppl in turing stan; do
        julia benchmarks/benchmark_slurm.jl ${dataset} ${ppl}
    done
done
