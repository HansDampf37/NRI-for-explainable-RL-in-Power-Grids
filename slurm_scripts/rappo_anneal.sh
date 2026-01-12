experiment_name=1201_rappo_with_anneal
export experiment_name
cd ..

# Create output directories
mkdir -p results/experiments/${experiment_name}/out

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=anneal_ragnn                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=08:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/ppo_baseline_batchjob.yaml -wd . -s 0 -j 0 --model-type RAGNN --experiment-name ${experiment_name}
EOF