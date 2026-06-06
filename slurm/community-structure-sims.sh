#!/usr/bin/env bash

#SBATCH --job-name=community-structure-sims
#SBATCH --array=0-9
#SBATCH --mail-user=pchodrow@middlebury.edu    
#SBATCH --mail-type=ALL                        
#SBATCH --mem-per-cpu=30gb
#SBATCH --partition=standard                   
#SBATCH --time=24:00:00        
#SBATCH --output=throughput/logs/community-structure/community_structure_sims_%a.out            

env/bin/python -u -m scripts.community-structure-sims --n_steps 1000 --k_reps 10 --job_id $SLURM_ARRAY_TASK_ID
