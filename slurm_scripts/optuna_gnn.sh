#!/bin/bash

experiment_name=gnn_optuna
cd ..

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=gnn_optuna                                           # Name of the job
#SBATCH --output=results/logs/gnn_optuna.%j.log                        # Output file
#SBATCH --error=results/logs/gnn_optuna_error.%j.log                   # Error file
#SBATCH --ntasks=1                                                      # Number of tasks
#SBATCH --cpus-per-task=111
#SBATCH --time=24:00:00                                                 # Max wall time (HH:MM:SS)
#SBATCH --mem=100G                                                      # Memory requirement
#SBATCH --partition=cpu,cpu_il                                          # Specify the GPU partition

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=\$(pwd) python training_scripts/train_ppo.py \\
    --file_path configs/ppo_gnn_optuna.yaml \\
    --workdir \$(pwd) \\
    --model-type GNN \\
    --job_id \${SLURM_JOB_ID}
EOF

