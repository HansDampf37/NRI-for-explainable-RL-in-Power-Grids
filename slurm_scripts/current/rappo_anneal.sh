experiment_name=1601_rappo_with_anneal_different_betas
export experiment_name
cd ../../

# Create output directories
mkdir -p results/experiments/${experiment_name}/out

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=anneal_ragnn                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/rappo_anneal.yaml -wd . -s 0 -j 0 --model-type RAGNN --experiment-name ${experiment_name}
EOF
