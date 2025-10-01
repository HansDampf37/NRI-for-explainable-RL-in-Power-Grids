sbatch <<'EOF'
#!/bin/bash

#SBATCH --job-name=train_baseline_gnn_ppo                                   # Name of the job
#SBATCH --output=out/train_baseline_gnn_ppo_v2.%j.log                       # Output file
#SBATCH --error=out/error_train_baseline_gnn_ppo_v2.%j.log                  # Error file
#SBATCH --ntasks=1                                                          # Number of tasks
#SBATCH --gres=gpu:1                                                        # Request 1 GPU
#SBATCH --time=24:00:00                                                     # Max wall time (HH:MM:SS)
#SBATCH --mem=5G                                                            # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                        # Specify the GPU partition gpu_mi300

cd ..

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python train_baseline.py baseline.model=gnn_ppo baseline.obs_space=graph_obs_space env.safe_max_rho=0.98 baseline.model.name=gnn-ppo-rho98-mazereward_uc3
EOF