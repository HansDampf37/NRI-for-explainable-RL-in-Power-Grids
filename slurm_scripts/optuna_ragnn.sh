#!/bin/bash

# Set experiment name and export variable for SLURM
experiment_name="0501_ragnn_hyperparameter_optimization"
export experiment_name
cd ..

# Ensure that the output directory exists
mkdir -p results/experiments/${experiment_name}/out

# Submit SLURM job
sbatch << EOF
#!/bin/bash

#SBATCH --job-name=ragnn_optuna                                           # Job name
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_optuna.%j.log                          # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/ragnn_optuna_error.%j.log                     # Error file
#SBATCH --ntasks=1                                                        # Number of tasks
#SBATCH --cpus-per-task=111                                                # Number of CPU cores per task
#SBATCH --time=72:00:00                                                   # Max wall time (HH:MM:SS)
#SBATCH --mem=160G                                                        # Memory requirement
#SBATCH --partition=cpu,cpu_il                                            # Partition to use

# Load the necessary module and activate conda environment
module load devel/miniforge
conda activate L2RPN

# Run the Python script
PYTHONPATH=$(pwd) python training_scripts/train_ppo.py \
    --file_path configs/ppo_ragnn_optuna.yaml \
    --workdir $(pwd) \
    --model-type RAGNN \
    --job_id \${SLURM_JOB_ID} \
    --experiment-name ${experiment_name}
EOF
