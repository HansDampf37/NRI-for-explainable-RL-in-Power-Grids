sbatch <<'EOF'
#!/bin/bash

#SBATCH --job-name=get_latent_edges                                      # Name of the job
#SBATCH --output=out/get_latent_edges.%j.log                             # Output file
#SBATCH --error=out/get_latent_edges.%j.log                              # Error file
#SBATCH --ntasks=1                                                       # Number of tasks
#SBATCH --gres=gpu:1                                                     # Request 1 GPU
#SBATCH --time=00:30:00                                                  # Max wall time (HH:MM:SS)
#SBATCH --mem=16G                                                        # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il #,gpu_mi300         # Specify the GPU partition

cd ..

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python nri/get_edge_probs.py
EOF
