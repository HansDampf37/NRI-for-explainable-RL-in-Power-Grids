sbatch <<'EOF'
#!/bin/bash

#SBATCH --job-name=gen_nri_dataset                                             # Name of the job
#SBATCH --output=out/gen_ds/gen_nri_dataset.%j.log                             # Output file
#SBATCH --error=out/gen_ds/error_gen_nri_dataset.%j.log                        # Error file
#SBATCH --ntasks=1                                                             # Number of tasks
#SBATCH --gres=gpu:1                                                           # Request 1 GPU
#SBATCH --time=03:00:00                                                        # Max wall time (HH:MM:SS)
#SBATCH --mem=32G                                                              # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il #,gpu_mi300               # Specify the GPU partition

cd ..

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python nri/create_dataset.py
EOF
