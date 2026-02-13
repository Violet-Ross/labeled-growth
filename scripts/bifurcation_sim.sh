#!/usr/bin/env bash
# slurm template for serial jobs
# Set SLURM options
#SBATCH --job-name=serial_test # Job name
#SBATCH --output=serial_test-%j.out
# Standard output and error log
#SBATCH --mail-user=username@middlebury.edu
# Where to send mail
#SBATCH --mail-type=NONE
# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=100mb
# Job memory request 
#SBATCH --partition=standard
# Partition (queue) 
#SBATCH --time=24:00:00
# Time limit hrs:min:sec 

srun python3 bifurcation_sim.py