#!/usr/bin/env bash

#SBATCH --job-name=community-synthetic-sims
#SBATCH --array=0-4
#SBATCH --mail-user=pchodrow@middlebury.edu    
#SBATCH --mail-type=ALL                        
#SBATCH --mem-per-cpu=30gb
#SBATCH --partition=standard                   
#SBATCH --time=24:00:00        
#SBATCH --output=throughput/logs/community-synthetic/community_synthetic_sims_%a.out            

env/bin/python -u -m scripts.community-synthetic-data --steps 100 --resolution 10 --runs 20 --job $SLURM_ARRAY_TASK_ID
