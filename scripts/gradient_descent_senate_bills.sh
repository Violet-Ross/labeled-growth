#!/usr/bin/env bash

# SLURM template for serial jobs

# Set SLURM options
#SBATCH --job-name=senate_bills      # Job name
#SBATCH --output=./throughput/senate_bills%j.out # Output file incorporating job ID
#SBATCH --partition=himem-long        # Partition (queue) 
#SBATCH --time=144:00:00             # Time limit hrs:min:sec
#SBATCH --mem=400G                 # Job memory request 
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fcataldo@middlebury.edu


# Print SLURM environment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 

# Start of job info
echo "Starting: "`date +"%D %T"` 

# Your calculations here
.venv/bin/python -m scripts.gradient_descent_senate_bills


# End of job info 
echo "Ending: "`date +"%D %T"`

