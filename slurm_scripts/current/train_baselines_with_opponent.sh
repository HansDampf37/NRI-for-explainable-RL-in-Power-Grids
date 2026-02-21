experiment_name=26_02_21_baselines_gnn_and_mlp_with_opponent
export experiment_name
cd ../../

# Create output directories
mkdir -p results/experiments/${experiment_name}/out

for seed in 0 1 2 3 4; do
sbatch << EOF
#!/bin/bash

#SBATCH --job-name=gnn_opp_s${seed}                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/gnn_baseline_opp_s${seed}.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_gnn_baseline_opp_s${seed}.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/rappo_anneal_with_opponent.yaml -wd . -s ${seed} -j 0 --model-type GNN --experiment-name ${experiment_name} --opponent
EOF

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=mlp_opp_s${seed}                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/mlp_baseline_opp_s${seed}.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_mlp_baseline_opp_s${seed}.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/rappo_anneal_with_opponent.yaml -wd . -s ${seed} -j 0 --model-type MLP --experiment-name ${experiment_name} --opponent
EOF
done
