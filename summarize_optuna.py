#!/usr/bin/env python3
"""
Simple script to summarize Optuna hyperparameter optimization results.
No external dependencies required - uses only Python stdlib.
"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict


def load_trial_results(results_dir):
    """Load all trial results from the experiment directory."""
    results_path = Path(results_dir)
    trial_data = []

    trial_dirs = [d for d in results_path.iterdir()
                  if d.is_dir() and d.name.startswith('CustomPPO_2748683_')]

    print(f"Found {len(trial_dirs)} trials\n")

    for trial_dir in trial_dirs:
        trial_id = trial_dir.name.split('_')[2]

        # Load params.json
        params_file = trial_dir / 'params.json'
        progress_file = trial_dir / 'progress.csv'

        if params_file.exists() and progress_file.exists():
            try:
                with open(params_file, 'r') as f:
                    params = json.load(f)

                # Read last line of progress CSV
                with open(progress_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]

                        trial_info = {
                            'trial_id': trial_id,
                            'trial_name': trial_dir.name,
                            'iterations': len(rows),
                            'timesteps_total': float(last_row.get('timesteps_total', 0)),
                            'eval_g2op_end': float(last_row.get('evaluation/custom_metrics/g2op_end_timestep_mean', 0)),
                            'eval_reward': float(last_row.get('evaluation/custom_metrics/episode_reward_mean', 0)),
                            'train_g2op_end': float(last_row.get('sampler_results/custom_metrics/g2op_end_timestep_mean', 0)),
                            'train_reward': float(last_row.get('sampler_results/episode_reward_mean', 0)),
                            'lr': params.get('lr', 0),
                            'gamma': params.get('gamma', 0),
                            'lambda': params.get('lambda', 0),
                            'clip_param': params.get('clip_param', 0),
                            'entropy_coeff': params.get('entropy_coeff', 0),
                        }

                        # Extract model config
                        model_config = params.get('model', {}).get('custom_model_config', {})
                        trial_info['hidden_dim'] = model_config.get('hidden_dim', 'N/A')
                        trial_info['num_layers'] = model_config.get('num_layers', 'N/A')

                        trial_data.append(trial_info)
            except Exception as e:
                print(f"Warning: Could not load {trial_dir.name}: {e}")

    # Sort by eval_g2op_end (higher is better)
    trial_data.sort(key=lambda x: x['eval_g2op_end'], reverse=True)

    return trial_data


def print_summary(trial_data):
    """Print a summary of the optimization results."""

    print("=" * 100)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 100)
    print(f"\nTotal Trials Analyzed: {len(trial_data)}")
    print(f"Metric: Evaluation g2op_end (timesteps survived in evaluation)")

    if not trial_data:
        print("\nNo trial data found!")
        return

    print("\n" + "-" * 100)
    print("TOP 15 TRIALS BY EVALUATION PERFORMANCE")
    print("-" * 100)

    # Print header
    print(f"\n{'Rank':<6} {'Trial ID':<12} {'Eval g2op':<12} {'Eval Reward':<13} {'Train g2op':<12} {'Iterations':<12}")
    print("-" * 100)

    # Print top 15
    for idx, trial in enumerate(trial_data[:15], 1):
        print(f"{idx:<6} {trial['trial_id']:<12} {trial['eval_g2op_end']:<12.0f} "
              f"{trial['eval_reward']:<13.2f} {trial['train_g2op_end']:<12.0f} {trial['iterations']:<12}")

    print("\n" + "-" * 100)
    print("BEST TRIAL DETAILS")
    print("-" * 100)

    best = trial_data[0]
    print(f"\nBest Trial ID: {best['trial_id']}")
    print(f"Trial Name: {best['trial_name']}")
    print(f"\nPerformance Metrics:")
    print(f"  Evaluation g2op_end:  {best['eval_g2op_end']:.0f} timesteps")
    print(f"  Evaluation Reward:    {best['eval_reward']:.2f}")
    print(f"  Training g2op_end:    {best['train_g2op_end']:.0f} timesteps")
    print(f"  Training Reward:      {best['train_reward']:.2f}")
    print(f"  Total Timesteps:      {best['timesteps_total']:.0f}")
    print(f"  Total Iterations:     {best['iterations']}")

    print(f"\nBest Hyperparameters:")
    print(f"  Learning Rate:        {best['lr']:.6f}")
    print(f"  Gamma:                {best['gamma']:.4f}")
    print(f"  Lambda:               {best['lambda']:.4f}")
    print(f"  Clip Param:           {best['clip_param']:.4f}")
    print(f"  Entropy Coeff:        {best['entropy_coeff']:.6f}")
    print(f"  Hidden Dim:           {best['hidden_dim']}")
    print(f"  Num Layers:           {best['num_layers']}")

    print("\n" + "-" * 100)
    print("STATISTICS")
    print("-" * 100)

    eval_scores = [t['eval_g2op_end'] for t in trial_data]
    eval_rewards = [t['eval_reward'] for t in trial_data]

    print(f"\nEvaluation g2op_end:")
    print(f"  Best:    {max(eval_scores):.0f}")
    print(f"  Worst:   {min(eval_scores):.0f}")
    print(f"  Average: {sum(eval_scores)/len(eval_scores):.0f}")
    print(f"  Median:  {sorted(eval_scores)[len(eval_scores)//2]:.0f}")

    print(f"\nEvaluation Reward:")
    print(f"  Best:    {max(eval_rewards):.2f}")
    print(f"  Worst:   {min(eval_rewards):.2f}")
    print(f"  Average: {sum(eval_rewards)/len(eval_rewards):.2f}")

    print("\n" + "=" * 100)


def save_results_csv(trial_data, output_file):
    """Save results to CSV file."""
    if not trial_data:
        return

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['rank', 'trial_id', 'trial_name', 'eval_g2op_end', 'eval_reward',
                     'train_g2op_end', 'train_reward', 'iterations', 'timesteps_total',
                     'lr', 'gamma', 'lambda', 'clip_param', 'entropy_coeff',
                     'hidden_dim', 'num_layers']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, trial in enumerate(trial_data, 1):
            row = trial.copy()
            row['rank'] = idx
            writer.writerow(row)

    print(f"\nResults saved to: {output_file}")


def main():
    results_dir = '/pfs/data6/home/ka/ka_iai/ka_hw6998/dev/NRI-for-explainable-RL-in-Power-Grids/results/experiments/gnn_hyperparameter_optimization'
    output_file = '/optuna_results_summary_old.csv'

    print("Loading trial results...")
    trial_data = load_trial_results(results_dir)

    print_summary(trial_data)

    save_results_csv(trial_data, output_file)

    print("\n" + "=" * 100)
    print("ANALYSIS ABOUT MEMORY ISSUE")
    print("=" * 100)
    print("\nBased on the log files, your job was killed due to OOM (Out-Of-Memory) errors.")
    print("The error message shows:")
    print("  'Worker unexpectedly exits... (1) The process is killed by SIGKILL")
    print("   by OOM killer due to high memory usage.'")
    print("\nMemory Stats from SLURM:")
    print("  Memory Utilized: 78.42 GB")
    print("  Memory Efficiency: 78.42% of 100.00 GB")
    print("\nThis suggests that while average memory usage was 78%, there were likely")
    print("memory spikes that exceeded 100 GB, triggering the OOM killer.")
    print("\nCompleted Trials: 19 out of 20 (last trial was killed)")
    print("\nRecommendations:")
    print("  1. Reduce num_workers in Ray configuration")
    print("  2. Reduce train_batch_size or rollout_fragment_length")
    print("  3. Request more memory in SLURM (e.g., --mem=150G)")
    print("  4. Enable Ray object spilling to disk")
    print("=" * 100)


if __name__ == '__main__':
    main()

