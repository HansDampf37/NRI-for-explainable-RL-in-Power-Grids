sbatch << 'EOF'
#!/bin/bash

#SBATCH --job-name=train_nri                               # Name of the job
#SBATCH --output=out/exp1/train_nri.%j.log                 # Output file
#SBATCH --error=out/exp1/error_train_nri.%j.log            # Error file
#SBATCH --ntasks=1                                                       # Number of tasks
#SBATCH --gres=gpu:1                                                     # Request 1 GPU
#SBATCH --time=48:00:00                                                  # Max wall time (HH:MM:SS)
#SBATCH --mem=64G                                                        # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il #,gpu_mi300         # Specify the GPU partition

cd ..

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python nri/train_nri.py nri=default
EOF
