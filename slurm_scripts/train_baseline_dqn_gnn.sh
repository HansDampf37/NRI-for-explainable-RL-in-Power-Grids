sbatch <<'EOF'
#!/bin/bash

#SBATCH --job-name=train_gnn_baseline                                         # Name of the job
#SBATCH --output=out/gnn_baseline/train_baseline_gnn.%j.log                   # Output file
#SBATCH --error=out/gnn_baseline/error_train_baseline_gnn_v2.%j.log           # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --gres=gpu:1                                                          # Request 1 GPU
#SBATCH --time=24:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=5G                                                              # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

cd ..

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python nri/agent/dqn/train_GNN_baseline.py
EOF