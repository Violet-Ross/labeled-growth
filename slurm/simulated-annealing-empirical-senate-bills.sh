#!/usr/bin/env bash

#SBATCH --job-name=simulated-annealing-senate
#SBATCH --array=0-9
#SBATCH --mail-user=fcataldo@middlebury.edu    
#SBATCH --mail-type=ALL                        
#SBATCH --mem-per-cpu=50gb
#SBATCH --partition=himem-long                   
#SBATCH --time=120:00:00        
#SBATCH --output=throughput/logs/simulated-annealing-empirical/senate_bills_%a.out            

.venv/bin/python -u -m scripts.simulated-annealing-empirical --data_set senate_bills --job_id $SLURM_ARRAY_TASK_ID