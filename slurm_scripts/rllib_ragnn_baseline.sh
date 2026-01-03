experiment_name=christmas_ppo_rllib
cd ..

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=ra_ray_ppo_baseline                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=03:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=100G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo_baseline.py -f configs/ppo_baseline_batchjob.yaml -wd . -s 0 -j 0 --model-type RAGNN
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=gnn_ray_ppo_baseline                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/gnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_gnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=03:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=100G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo_baseline.py -f configs/ppo_baseline_batchjob.yaml -wd . -s 0 -j 0 --model-type GNN
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=mlp_ray_ppo_baseline                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/mlp_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_mlp_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=03:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=100G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo_baseline.py -f configs/ppo_baseline_batchjob.yaml -wd . -s 0 -j 0 --model-type MLP
EOF