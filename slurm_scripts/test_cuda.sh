sbatch <<'EOF'
#!/bin/bash

#SBATCH --job-name=test_cuda                    # Name of the job
#SBATCH --output=out/test_cuda.%j.log           # Output file (with job ID)
#SBATCH --error=out/test_cuda.%j.log            # Error file (with job ID)
#SBATCH --ntasks=1                              # Number of tasks (1 for a single GPU job)
#SBATCH --gres=gpu:1                            # Request 2 GPU
#SBATCH --time=00:01:00                         # Max wall time (HH:MM:SS)
#SBATCH --mem=200mb                             # Memory requirement (adjust as needed)
#SBATCH --partition=gpu_h100,gpu_mi300,gpu_a100_il,gpu_h100_il              # GPU partition

cd ..

module load devel/miniforge
conda activate RL

python test_cuda.py
EOF