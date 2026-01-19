experiment_name=1901_rainbow_baselines
export experiment_name
cd ../../

# Create output directories
mkdir -p results/experiments/${experiment_name}/out

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=gnn_rb_opp                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/rappo_anneal_with_opponent.yaml -wd . -s 0 -j 0 --model-type GNN --experiment-name ${experiment_name} --opponent
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=gnn_rb                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/rappo_anneal.yaml -wd . -s 0 -j 0 --model-type GNN --experiment-name ${experiment_name}
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=mlp_rb_opp                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/rappo_anneal_with_opponent.yaml -wd . -s 0 -j 0 --model-type MLP --experiment-name ${experiment_name} --opponent
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=mlp_rb                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/rappo_anneal.yaml -wd . -s 0 -j 0 --model-type MLP --experiment-name ${experiment_name}
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=nri_opp_rb                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/rappo_anneal_with_opponent.yaml -wd . -s 0 -j 0 --model-type NRIGNN --experiment-name ${experiment_name} --opponent
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=nri_rb                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/rappo_anneal.yaml -wd . -s 0 -j 0 --model-type NRIGNN --experiment-name ${experiment_name}
EOF
