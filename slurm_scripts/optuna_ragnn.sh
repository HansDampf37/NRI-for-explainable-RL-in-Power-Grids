#!/bin/bash

experiment_name=ragnn_optuna
cd ..

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=ragnn_optuna                                         # Name of the job
#SBATCH --output=results/logs/ragnn_optuna.%j.log                      # Output file
#SBATCH --error=results/logs/ragnn_optuna_error.%j.log                 # Error file
#SBATCH --ntasks=1                                                      # Number of tasks
#SBATCH --gres=gpu:1                                                    # Request 1 GPU
#SBATCH --time=24:00:00                                                 # Max wall time (HH:MM:SS)
#SBATCH --mem=20G                                                       # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                   # Specify the GPU partition

module load devel/miniforge
conda activate RL

PYTHONPATH=\$(pwd) python training_scripts/train_ppo.py \\
    --file_path configs/ppo_ragnn_optuna.yaml \\
    --workdir \$(pwd) \\
    --model-type RAGNN \\
    --job_id \${SLURM_JOB_ID}
EOF

