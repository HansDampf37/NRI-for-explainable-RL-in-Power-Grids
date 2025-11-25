#!/bin/bash

priors=(1.0 0.9 0.8)
experiment_name=compare_encoder
cd ..

for p in "${priors[@]}"; do
sbatch << EOF
#!/bin/bash

#SBATCH --job-name=rappo_${p}_graphormer                                           # Name of the job
#SBATCH --output=data/experiments/${experiment_name}/relations_aware/ppo/graphormer/rappo_${p}.%j.log                  # Output file
#SBATCH --error=data/experiments/${experiment_name}/relations_aware/ppo/graphormer/error_rappo_${p}.%j.log             # Error file
#SBATCH --ntasks=1                                                      # Number of tasks
#SBATCH --gres=gpu:1                                                    # Request 1 GPU
#SBATCH --time=16:00:00                                                 # Max wall time (HH:MM:SS)
#SBATCH --mem=64G                                                       # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il #,gpu_mi300        # Specify the GPU partition

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/train_relations_aware_ppo.py rl.model.name_suffix=graphormer rl.model.prior_for_graph_edges_existing=${p} rl.model.use_graphormer=true experiment_name=${experiment_name}
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=rappo_${p}_nri_encoder                                           # Name of the job
#SBATCH --output=data/experiments/${experiment_name}/relations_aware/ppo/rappo_${p}_nri_encoder.%j.log                  # Output file
#SBATCH --error=data/experiments/${experiment_name}/relations_aware/ppo/error_rappo_${p}_nri_encoder.%j.log             # Error file
#SBATCH --ntasks=1                                                      # Number of tasks
#SBATCH --gres=gpu:1                                                    # Request 1 GPU
#SBATCH --time=16:00:00                                                 # Max wall time (HH:MM:SS)
#SBATCH --mem=64G                                                       # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il #,gpu_mi300        # Specify the GPU partition

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/train_relations_aware_ppo.py rl.model.name_suffix=nri_encoder rl.model.prior_for_graph_edges_existing=${p} rl.model.use_graphormer=false experiment_name=${experiment_name}
EOF
done