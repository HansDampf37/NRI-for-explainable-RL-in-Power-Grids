#!/bin/bash

experiment_name=compare_dqn_explorations

sbatch <<'EOF'
#!/bin/bash

#SBATCH --job-name=dqn_mlp_baseline                                               # Name of the job
#SBATCH --output=data/${experiment_name}/out/relations_unaware/dqn/train_baseline_mlp.%j.log            # Output file
#SBATCH --error=data/${experiment_name}/out/relations_unaware/dqn/error_train_baseline_mlp.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --gres=gpu:1                                                          # Request 1 GPU
#SBATCH --time=24:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=10G                                                             # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

cd ..

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/train_mlp_dqn_baseline.py experiment_name=${experiment_name} rl.model.exploration=Softmax rl.model.exploration=softmax
EOF

sbatch <<'EOF'
#!/bin/bash

#SBATCH --job-name=dqn_mlp_baseline                                               # Name of the job
#SBATCH --output=data/${experiment_name}/out/relations_unaware/dqn/train_baseline_mlp.%j.log            # Output file
#SBATCH --error=data/${experiment_name}/out/relations_unaware/dqn/error_train_baseline_mlp.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --gres=gpu:1                                                          # Request 1 GPU
#SBATCH --time=24:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=10G                                                              # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

cd ..

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/train_mlp_dqn_baseline.py experiment_name=${experiment_name} rl.model.name_suffix=epsilon_greedy rl.model.exploration=epsilon_greedy
EOF
