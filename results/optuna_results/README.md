# Optuna Results

This directory contains CSV files with detailed results from Optuna hyperparameter optimization runs.

## File Naming Convention

Files are named as: `{experiment_name}_{timestamp}.csv`

For example: `gnn_hyperparameter_optimization_2026-01-04_14-30-45.csv`

## CSV Content

Each CSV file contains the following information for all trials:

### Trial Information
- `trial_id`: Sequential trial number
- `trial_name`: Unique trial identifier
- `status`: SUCCESS or FAILED

### Hyperparameters (for GNN models)
- `hidden_dim`: Hidden dimension for GNN layers
- `out_dim`: Output dimension of GNN
- `num_layers`: Number of GNN layers
- `residual`: Whether residual connections are used

### Metrics
- `grid2op_end_mean`: Average episode length on evaluation
- `corrected_ep_len_mean`: Corrected episode length mean
- `episode_reward_mean`: Mean episode reward
- `timesteps_total`: Total training timesteps
- `training_iteration`: Number of training iterations

### Error Information
- `error`: Error message (only present for failed trials)

## Terminal Output

When an Optuna optimization completes, you will see:

1. **Summary Statistics**: Total trials, successful/failed counts
2. **Top 5 Trials**: Best performing trials sorted by the optimization metric
3. **All Trials Summary**: Complete table of all trials and their results
4. **Best Trial**: Detailed information about the best performing trial with its hyperparameters
5. **CSV Location**: Path to the saved CSV file for further analysis

## Usage

After running an Optuna optimization (e.g., via `slurm_scripts/optuna_gnn.sh`), check:

1. The terminal output or SLURM log file for the summary
2. This directory for the detailed CSV results

You can load and analyze the CSV files using pandas:

```python
import pandas as pd

df = pd.read_csv('gnn_hyperparameter_optimization_2026-01-04_14-30-45.csv')

# View best trials
best_trials = df[df['status'] == 'SUCCESS'].nlargest(5, 'grid2op_end_mean')
print(best_trials)

# Analyze hyperparameter correlations
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=df, x='hidden_dim', y='grid2op_end_mean', hue='num_layers')
plt.show()
```

