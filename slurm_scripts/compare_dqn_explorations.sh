#!/bin/bash

experiment_name=compare_dqn_explorations
cd ..

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=dqn_gnn_baseline_softmax                                               # Name of the job
#SBATCH --output=data/experiments/${experiment_name}/out/dqn_gnn_baseline_softmax.%j.log            # Output file
#SBATCH --error=data/experiments/${experiment_name}/out/dqn_gnn_baseline_softmax_error.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --gres=gpu:1                                                          # Request 1 GPU
#SBATCH --time=8:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=10G                                                             # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/train_gnn_dqn_baseline.py experiment_name=${experiment_name} rl.dqn.exploration=Softmax rl.model.name_suffix=softmax rl.train.timesteps=200000
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=dqn_gnn_baseline_epsilon_greedy                                               # Name of the job
#SBATCH --output=data/experiments/${experiment_name}/out/dqn_gnn_baseline_eps_greedy.%j.log            # Output file
#SBATCH --error=data/experiments/${experiment_name}/out/dqn_gnn_baseline_eps_greedy_error.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --gres=gpu:1                                                          # Request 1 GPU
#SBATCH --time=8:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=10G                                                              # Memory requirement
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate RL

python test_cuda.py && PYTHONPATH=$(pwd) python training_scripts/train_gnn_dqn_baseline.py experiment_name=${experiment_name} rl.dqn.exploration=epsilon_greedy rl.model.name_suffix=epsilon_greedy rl.train.timesteps=200000
EOF
