#!/bin/bash

experiment_name=optuna
cd ..

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=optuna                                               # Name of the job
#SBATCH --output=data/experiments/${experiment_name}/out/optuna.%j.log            # Output file
#SBATCH --error=data/experiments/${experiment_name}/out/optuna.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --gres=gpu:1                                                          # Request 1 GPU
#SBATCH --time=17:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=10G                                                             # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/optuna_rappo.py experiment_name=${experiment_name} timeout="14:00:00"
EOF
