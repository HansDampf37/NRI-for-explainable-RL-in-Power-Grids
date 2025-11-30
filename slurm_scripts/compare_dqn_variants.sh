#!/bin/bash

priors=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)
experiment_name=compare_dqn_variants
cd ..

for p in "${priors[@]}"; do
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=radqn_${p}
#SBATCH --output=data/experiments/${experiment_name}/out/radqn_${p}.%j.log
#SBATCH --error=data/experiments/${experiment_name}/out/error_radqn_${p}.%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/train_relations_aware_dqn.py rl.model.name_suffix=${p} rl.model.prior_for_graph_edges_existing=${p} rl.model.use_graphormer=true experiment_name=${experiment_name}
EOF
done


sbatch << EOF
#!/bin/bash

#SBATCH --job-name=dqn_mlp_baseline                                           # Name of the job
#SBATCH --output=data/experiments/${experiment_name}/out/dqn_nlp_baseline.%j.log            # Output file
#SBATCH --error=data/experiments/${experiment_name}/out/dqn_nlp_baseline_error.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --gres=gpu:1                                                          # Request 1 GPU
#SBATCH --time=8:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=10G                                                             # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/train_mlp_dqn_baseline.py experiment_name=${experiment_name}
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=dqn_gnn_baseline                                           # Name of the job
#SBATCH --output=data/experiments/${experiment_name}/out/dqn_gnn_baseline.%j.log            # Output file
#SBATCH --error=data/experiments/${experiment_name}/out/dqn_gnn_baseline_error.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --gres=gpu:1                                                          # Request 1 GPU
#SBATCH --time=8:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=10G                                                             # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/train_gnn_dqn_baseline.py experiment_name=${experiment_name}
EOF