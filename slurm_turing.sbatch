#!/bin/bash
#
#SBATCH --job-name=static_turing_benchmarks
#SBATCH --output=static_turing_benchmarks.%A_%a.out
#SBATCH --error=static_turing_benchmarks.%A_%a.error
#
#SBATCH --time=20:00:00
#SBATCH --array=0-8
#SBATCH -p simulation12
#SBATCH --mem 20G 
#SBATCH --cpus-per-task 1
#SBATCH -D /clusterFS/home/user/martint/git/DynamicPPL_NeurIPS 

ARGS=(high_dim_gauss gauss_unknown h_poisson hmm_semisup naive_bayes logistic_reg sto_volatility lda lda_unvectorized)

srun julia_1.4.1.afs benchmarks/benchmark_slurm.jl ${ARGS[$SLURM_ARRAY_TASK_ID]} turing
