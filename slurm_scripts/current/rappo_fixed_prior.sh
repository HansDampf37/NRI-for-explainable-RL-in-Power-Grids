experiment_name=2002_rappo_fixed_prior
export experiment_name
cd ../../

# Create output directories
mkdir -p results/experiments/${experiment_name}/out

# Submit 5 jobs with different seeds
for seed in 0 1 2 3 4; do
sbatch << EOF
#!/bin/bash

#SBATCH --job-name=fixed_prior_s${seed}                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo_s${seed}.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo_s${seed}.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=111                                                    # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python training_scripts/train_ppo.py -f configs/rappo_fixed_prior.yaml -wd . -s ${seed} -j 0 --model-type RAGNN --experiment-name ${experiment_name}
EOF
done

