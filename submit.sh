#!/bin/bash

sbatch --export=ALL,JULIA_DEPOT_PATH='/clusterFS/home/user/martint/julia_depo_2' slurm_turing.sbatch
sbatch --export=ALL,JULIA_DEPOT_PATH='/clusterFS/home/user/martint/julia_depo_2' slurm_stan.sbatch
sbatch --export=ALL,JULIA_DEPOT_PATH='/clusterFS/home/user/martint/julia_depo_2' slurm_dynamic.sbatch
