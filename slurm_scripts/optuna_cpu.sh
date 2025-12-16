#!/bin/bash

experiment_name=optuna
cd ..

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=optuna                                                     # Name of the job
#SBATCH --output=data/experiments/${experiment_name}/out/optuna.%j.log        # Output file
#SBATCH --error=data/experiments/${experiment_name}/out/optuna_error.%j.log         # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --time=8:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=10G                                                             # Memory requirement
#SBATCH --partition=cpu_il,cpu                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate RL

PYTHONPATH=$(pwd) python training_scripts/optuna_rappo.py experiment_name=${experiment_name} timeout="06:00:00"
EOF
