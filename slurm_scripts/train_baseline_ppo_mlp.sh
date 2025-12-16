experiment_name=improve_baseline
cd ..

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=ppo_mlp_baseline                                           # Name of the job
#SBATCH --output=data/experiments/${experiment_name}/out/train_baseline_mlp.%j.log            # Output file
#SBATCH --error=data/experiments/${experiment_name}/out/error_train_baseline_mlp.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --gres=gpu:1                                                          # Request 1 GPU
#SBATCH --time=12:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=10G                                                             # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/train_mlp_ppo_baseline.py
EOF