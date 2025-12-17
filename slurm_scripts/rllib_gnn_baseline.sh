experiment_name=ray_gnn
cd ..

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=ray_ppo_mlp_baseline                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/train_ray_baseline_ppo_mlp.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_train_ray_baseline_ppo_mlp.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=20                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=100G                                                            # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo_baseline.py -f configs/ppo_baseline_batchjob.yaml -wd . -s 0 -j 0
EOF