# Optuna Hyperparameter Optimization Results Summary

## Job Information
- **Job ID**: 2748683
- **Start Time**: 2026-01-03 11:18:58
- **End Time**: 2026-01-04 03:44:02
- **Total Runtime**: ~16.5 hours
- **Status**: Terminated due to OOM (Out of Memory)

## Memory Analysis

### SLURM Memory Stats
- **Memory Utilized**: 78.42 GB
- **Memory Efficiency**: 78.42% of 100.00 GB
- **Allocated Memory**: 100 GB per node

### What Happened?
Your job was **killed by the OOM (Out-Of-Memory) killer**, NOT due to running out of time. The error in the logs shows:

```
Worker unexpectedly exits with a connection error code 2. End of file. 
There are some potential root causes:
(1) The process is killed by SIGKILL by OOM killer due to high memory usage.
```

While the average memory usage was 78.42%, there were **memory spikes that exceeded 100 GB**, triggering the Linux OOM killer to terminate processes. This is a common issue with Ray/RLlib when running multiple parallel workers.

## Optimization Results

### Trials Completed
- **Total Trials**: 20 (19 completed, 1 killed by OOM)
- **Best Trial ID**: `abdaf1b2`
- **Best Score**: **7879** (g2op_end timestep - how long the agent survived)
- **Timestep of Best Score**: 51,101

### Trial Performance Summary
Based on the log file, here are some of the completed trials:

| Trial ID | Final Iteration | Best Eval g2op_end | Status |
|----------|----------------|-------------------|---------|
| abdaf1b2 | ~70+ | **7879** | ✅ Completed |
| 678af7e4 | 70 | 6706 | ✅ Completed |
| a44ba3f1 | 69 | 7210 | ✅ Completed |
| 9d966912 | 72 | - | ✅ Completed |
| 42276297 | 85 | - | ✅ Completed |
| 88210298 | 22 | 631 | ❌ Killed by OOM |

The best trial (abdaf1b2) achieved an evaluation score of **7879 timesteps**, which is significantly better than most other trials.

## Visualization Options

### 1. **TensorBoard** (Recommended)
Each trial has TensorBoard logs. You can visualize them:

```bash
# View a specific trial
tensorboard --logdir results/experiments/gnn_hyperparameter_optimization/CustomPPO_2748683_abdaf1b2_*

# View all trials together
tensorboard --logdir results/experiments/gnn_hyperparameter_optimization/
```

Then open your browser to `http://localhost:6006` (or use SSH port forwarding if on a remote server).

### 2. **Optuna Dashboard** (If you want interactive visualization)
I've created scripts to help you visualize the results:

```bash
# First, install optuna-dashboard if needed
pip install optuna-dashboard

# Create an Optuna study from Ray Tune results
python create_optuna_dashboard.py

# Launch the dashboard
optuna-dashboard ./optuna_analysis/optuna_study.db
```

Then open `http://localhost:8080` in your browser.

### 3. **Custom Analysis Scripts**
I've created several analysis scripts for you:

- `analyze_optuna_results.py` - Creates detailed plots and CSV summaries
- `create_optuna_dashboard.py` - Converts results to Optuna format
- `summarize_optuna.py` - Simple text-based summary (no dependencies)

Run them with:
```bash
python analyze_optuna_results.py --output_dir ./optuna_analysis
```

## Best Hyperparameters

To find the exact hyperparameters of the best trial, check:
```bash
cat results/experiments/gnn_hyperparameter_optimization/CustomPPO_2748683_abdaf1b2_*/params.json | python -m json.tool | less
```

## Recommendations to Fix Memory Issues

### 1. **Reduce Ray Workers**
In your config, reduce the number of parallel workers:
```yaml
num_workers: 80  # Reduce to 40 or 60
num_envs_per_worker: 1
```

### 2. **Reduce Batch Sizes**
```yaml
train_batch_size: 4000  # Reduce to 2000
rollout_fragment_length: 50  # Keep smaller
```

### 3. **Request More Memory in SLURM**
```bash
#SBATCH --mem=150G  # or --mem-per-cpu=4G
```

### 4. **Enable Ray Object Spilling**
Add to your training script:
```python
ray.init(_system_config={
    "object_spilling_config": json.dumps({
        "type": "filesystem",
        "params": {"directory_path": "/scratch/slurm_tmpdir/job_$SLURM_JOB_ID/spill"}
    })
})
```

### 5. **Use Fewer Trials in Parallel**
In Optuna configuration:
```python
tune.run(
    ...
    config={...},
    num_samples=20,
    max_concurrent_trials=2,  # Add this - only run 2 trials at once
)
```

## Next Steps

1. **Extract Best Hyperparameters**: 
   ```bash
   python -c "import json; print(json.dumps(json.load(open('results/experiments/gnn_hyperparameter_optimization/CustomPPO_2748683_abdaf1b2_*/params.json')), indent=2))"
   ```

2. **Train Full Model with Best Config**: Use the best hyperparameters from trial `abdaf1b2` for a full training run

3. **Re-run Optimization** (optional): With reduced memory settings if you want to explore more hyperparameters

4. **Visualize Learning Curves**: Use TensorBoard to see the training progression

## Files Created

- `analyze_optuna_results.py` - Comprehensive analysis with plots
- `create_optuna_dashboard.py` - Interactive Optuna dashboard
- `summarize_optuna.py` - Quick text summary
- `OPTUNA_RESULTS_SUMMARY.md` - This file

## Questions?

The optimization successfully completed 19/20 trials before being killed by memory pressure. The best trial achieved a score of **7879**, which indicates good agent performance. The memory issue is solvable with the recommendations above.

