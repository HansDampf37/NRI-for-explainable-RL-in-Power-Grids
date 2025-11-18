#!/bin/bash

priors=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)

for p in "${priors[@]}"; do
sbatch << EOF
#!/bin/bash

#SBATCH --job-name=radqn_${p}                                           # Name of the job
#SBATCH --output=out/relations_aware/radqn_${p}.%j.log                  # Output file
#SBATCH --error=out/relations_aware/error_radqn_${p}.%j.log             # Error file
#SBATCH --ntasks=1                                                      # Number of tasks
#SBATCH --gres=gpu:1                                                    # Request 1 GPU
#SBATCH --time=48:00:00                                                 # Max wall time (HH:MM:SS)
#SBATCH --mem=64G                                                       # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il #,gpu_mi300        # Specify the GPU partition

cd ..

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=\$(pwd) python nri/agent/dqn/train_RADQN.py rl.model.name_suffix=${p} rl.model.prior_for_graph_edges_existing=${p} rl.model.use_graphormer=true
EOF
done
