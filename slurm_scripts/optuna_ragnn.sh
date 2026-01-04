#!/bin/bash

experiment_name=ragnn_optuna
cd ..

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=ragnn_optuna                                         # Name of the job
#SBATCH --output=results/logs/ragnn_optuna.%j.log                      # Output file
#SBATCH --error=results/logs/ragnn_optuna_error.%j.log                 # Error file
#SBATCH --ntasks=1                                                      # Number of tasks
#SBATCH --cpus-per-task=111
#SBATCH --time=72:00:00                                                 # Max wall time (HH:MM:SS)
#SBATCH --mem=160G                                                      # Memory requirement
#SBATCH --partition=cpu,cpu_il                                          # Specify the CPU partition

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=\$(pwd) python training_scripts/train_ppo.py \\
    --file_path configs/ppo_ragnn_optuna.yaml \\
    --workdir \$(pwd) \\
    --model-type RAGNN \\
    --job_id \${SLURM_JOB_ID}
EOF

