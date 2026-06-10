#!/usr/bin/env bash

#SBATCH --job-name=simulated-annealing-highschool-class
#SBATCH --array=0-9
#SBATCH --mail-user=fcataldo@middlebury.edu    
#SBATCH --mail-type=ALL                        
#SBATCH --mem-per-cpu=125gb
#SBATCH --partition=himem-long                   
#SBATCH --time=120:00:00        
#SBATCH --output=throughput/logs/simulated-annealing-empirical/highschool-class_%a.out            

.venv/bin/python -u -m scripts.simulated-annealing-empirical --data_set highschool_class --job_id $SLURM_ARRAY_TASK_ID