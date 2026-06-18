#!/usr/bin/env bash

#SBATCH --job-name=spectral-senate
#SBATCH --mail-user=fcataldo@middlebury.edu    
#SBATCH --mail-type=ALL                        
#SBATCH --mem-per-cpu=20gb
#SBATCH --partition=standard               
#SBATCH --time=24:00:00     
#SBATCH --output=throughput/logs/spectral-clustering-empirical/senate_bills_%a.out            

.venv/bin/python -u -m scripts.spectral-clustering-empirical --data_set senate_bills --job_id 0