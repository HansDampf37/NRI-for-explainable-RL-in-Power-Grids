experiment_name=2601_compute_metrics
export experiment_name
cd ../../

# Create output directories
mkdir -p results/experiments/${experiment_name}/out

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=compute_metrics                                           # Name of the job
#SBATCH --output=results/experiments/${experiment_name}/out/ragnn_ppo.%j.log            # Output file
#SBATCH --error=results/experiments/${experiment_name}/out/error_ragnn_ppo.%j.log       # Error file
#SBATCH --ntasks=1                                                            # Number of tasks
#SBATCH --cpus-per-task=1                                                     # Number of CPU cores per task
#SBATCH --time=20:00:00                                                       # Max wall time (HH:MM:SS)
#SBATCH --mem=200G                                                            # Memory requirement
#SBATCH --partition=cpu,cpu_il                          # Specify the GPU partition gpu_mi300

module load devel/miniforge
conda activate L2RPN

PYTHONPATH=$(pwd) python src/ra_agents/analyze_latent_graphs/visualize_rappo_posterior.py
EOF
