#!/usr/bin/env bash
# slurm template for serial jobs
# Set SLURM options
#SBATCH --job-name=SEM_convergene # Job name
#SBATCH --output=serial_test-%j.out
# Standard output and error log
#SBATCH --mail-user=username@middlebury.edu
# Where to send mail
#SBATCH --mail-type=NONE
# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=12G
# Job memory request 
#SBATCH --partition=standard
# Partition (queue) 
#SBATCH --time=48:00:00
# Time limit hrs:min:sec 

# Run python script
cd ~/labeled-growth
srun python3 -m scripts.SEM_convergence_data_ADA