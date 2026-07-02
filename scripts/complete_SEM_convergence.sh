#!/usr/bin/env bash
# slurm template for a job array (one task per parameter set)
# Set SLURM options
#SBATCH --job-name=SEM_convergence     # Job name
#SBATCH --output=serial_test-%A_%a.out # Standard output and error log (%A=job id, %a=array index)
#SBATCH --mail-user=username@middlebury.edu
# Where to send mail
#SBATCH --mail-type=NONE
# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=12G
# Job memory request
#SBATCH --partition=gpu-standard
# Partition (queue)
#SBATCH --time=48:00:00
# Time limit hrs:min:sec
#SBATCH --array=1-40
# Job array: one task per parameter set (g1..g40). Each task runs
# 20 independent replicates for its assigned parameter set.

# Run python script
cd ~/labeled-growth

source ~/venvs/lab-grow/bin/activate

srun python3 -m scripts.complete_SEM_convergence
