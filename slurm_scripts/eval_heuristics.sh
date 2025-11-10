sbatch <<'EOF'
#!/bin/bash

#SBATCH --job-name=evaluate_heuristics                                   # Name of the job
#SBATCH --output=out/evaluate_heuristics.%j.log                             # Output file
#SBATCH --error=out/evaluate_heuristics.%j.log                              # Error file
#SBATCH --ntasks=1                                                       # Number of tasks
#SBATCH --gres=gpu:0                                                     # Request 1 GPU
#SBATCH --time=02:00:00                                                  # Max wall time (HH:MM:SS)
#SBATCH --mem=16G                                                        # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il #,gpu_mi300         # Specify the GPU partition

cd ..

module load devel/miniforge
conda activate RL

PYTHONPATH=$(pwd) python baselines/evaluate_heuristic_agents.py
EOF
