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
#SBATCH --mem=2G
# Job memory request 
#SBATCH --partition=standard
# Partition (queue) 
#SBATCH --time=24:00:00
# Time limit hrs:min:sec 

cd /home/vross/labeled-growth
export PYTHONPATH=/home/vross/labeled-growth:$PYTHONPATH
python3 scripts/SEM_explanatory_sim.py