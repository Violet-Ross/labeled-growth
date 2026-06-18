#!/usr/bin/env bash

#SBATCH --job-name=modularity-primaryschool
#SBATCH --mail-user=fcataldo@middlebury.edu    
#SBATCH --mail-type=ALL                        
#SBATCH --mem-per-cpu=20gb
#SBATCH --partition=standard               
#SBATCH --time=24:00:00     
#SBATCH --output=throughput/logs/modularity-empirical/primaryschool_%a.out            

.venv/bin/python -u -m scripts.modularity-empirical --data_set primaryschool_class --job_id 0