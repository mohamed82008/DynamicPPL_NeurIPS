#!/bin/bash
#
#SBATCH --job-name=dynamic_turing_benchmarks
#SBATCH --output=dynamic_turing_benchmarks.%A_%a.out
#SBATCH --error=dynamic_turing_benchmarks.%A_%a.error
#
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH -p simulation12
#SBATCH --mem 40G
#SBATCH --cpus-per-task 1
#SBATCH -D /clusterFS/home/user/martint/git/DynamicPPL_NeurIPS 

ARGS=(ibp crp changepoint demo_cflow)

srun julia_1.4.1.afs benchmarks/benchmark_slurm.jl ${ARGS[$SLURM_ARRAY_TASK_ID]} turing
