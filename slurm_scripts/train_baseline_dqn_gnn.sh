sbatch <<'EOF'
#!/bin/bash

#SBATCH --job-name=dqn_gnn_baseline                                               # Name of the job
#SBATCH --output=out/relations_unaware/dqn/train_baseline_gnn.%j.log            # Output file
#SBATCH --error=out/relations_unaware/dqn/error_train_baseline_gnn.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --gres=gpu:1                                                          # Request 1 GPU
#SBATCH --time=48:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=10G                                                             # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

cd ..

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python ra_agents/dqn/train_GNN_baseline.py
EOF