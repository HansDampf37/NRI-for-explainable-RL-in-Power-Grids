#!/usr/bin/env python3
"""
Script to analyze and visualize Optuna hyperparameter optimization results from Ray Tune.
This script reads the trial results and creates interactive visualizations.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_trial_results(results_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load all trial results from the experiment directory.

    Args:
        results_dir: Path to the results directory

    Returns:
        DataFrame with trial results and dict with trial metadata
    """
    results_path = Path(results_dir)
    trial_data = []
    trial_metadata = {}

    # Find all trial directories
    trial_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('CustomPPO_2748683_')]

    print(f"Found {len(trial_dirs)} trials")

    for trial_dir in trial_dirs:
        trial_id = trial_dir.name.split('_')[2]  # Extract trial ID

        # Load params.json
        params_file = trial_dir / 'params.json'
        if params_file.exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
                trial_metadata[trial_id] = params

        # Load progress.csv if it exists
        progress_file = trial_dir / 'progress.csv'
        if progress_file.exists():
            try:
                df = pd.read_csv(progress_file)
                if not df.empty:
                    # Get the last row (final results)
                    last_row = df.iloc[-1]

                    trial_info = {
                        'trial_id': trial_id,
                        'trial_name': trial_dir.name,
                        'iterations': len(df),
                        'timesteps_total': last_row.get('timesteps_total', 0),
                        'eval_g2op_end': last_row.get('evaluation/custom_metrics/g2op_end_timestep_mean', np.nan),
                        'eval_reward': last_row.get('evaluation/custom_metrics/episode_reward_mean', np.nan),
                        'train_g2op_end': last_row.get('sampler_results/custom_metrics/g2op_end_timestep_mean', np.nan),
                        'train_reward': last_row.get('sampler_results/episode_reward_mean', np.nan),
                        'train_ep_duration': last_row.get('sampler_results/custom_metrics/episode_duration_mean', np.nan),
                        'time_total_s': last_row.get('time_total_s', 0),
                    }

                    # Extract hyperparameters from params
                    if trial_id in trial_metadata:
                        params = trial_metadata[trial_id]
                        trial_info['lr'] = params.get('lr', np.nan)
                        trial_info['gamma'] = params.get('gamma', np.nan)
                        trial_info['lambda'] = params.get('lambda', np.nan)
                        trial_info['clip_param'] = params.get('clip_param', np.nan)
                        trial_info['entropy_coeff'] = params.get('entropy_coeff', np.nan)
                        trial_info['vf_loss_coeff'] = params.get('vf_loss_coeff', np.nan)

                        # GNN-specific parameters
                        model_config = params.get('model', {}).get('custom_model_config', {})
                        trial_info['hidden_dim'] = model_config.get('hidden_dim', np.nan)
                        trial_info['num_layers'] = model_config.get('num_layers', np.nan)
                        trial_info['aggregation'] = model_config.get('aggregation', 'unknown')

                    trial_data.append(trial_info)
            except Exception as e:
                print(f"Warning: Could not load progress for {trial_dir.name}: {e}")

    df = pd.DataFrame(trial_data)

    # Sort by eval_g2op_end (higher is better)
    if 'eval_g2op_end' in df.columns:
        df = df.sort_values('eval_g2op_end', ascending=False)

    return df, trial_metadata


def plot_trial_performance(df: pd.DataFrame, output_dir: str):
    """Create performance comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Best eval score per trial
    ax1 = axes[0, 0]
    top_n = min(20, len(df))
    df_top = df.head(top_n)
    ax1.barh(range(top_n), df_top['eval_g2op_end'].values)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(df_top['trial_id'].values, fontsize=8)
    ax1.set_xlabel('Evaluation g2op_end (timesteps survived)')
    ax1.set_title(f'Top {top_n} Trials by Evaluation Performance')
    ax1.invert_yaxis()

    # 2. Eval vs Train performance
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['train_g2op_end'], df['eval_g2op_end'],
                         c=df['iterations'], cmap='viridis', alpha=0.6, s=100)
    ax2.set_xlabel('Train g2op_end')
    ax2.set_ylabel('Eval g2op_end')
    ax2.set_title('Train vs Eval Performance')
    ax2.plot([0, df['train_g2op_end'].max()], [0, df['train_g2op_end'].max()],
             'r--', alpha=0.5, label='Perfect correlation')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Iterations')

    # 3. Reward comparison
    ax3 = axes[1, 0]
    ax3.scatter(df['train_reward'], df['eval_reward'], alpha=0.6, s=100)
    ax3.set_xlabel('Train Reward')
    ax3.set_ylabel('Eval Reward')
    ax3.set_title('Train vs Eval Reward')

    # 4. Performance over time
    ax4 = axes[1, 1]
    ax4.scatter(df['timesteps_total'], df['eval_g2op_end'], alpha=0.6, s=100)
    ax4.set_xlabel('Total Timesteps')
    ax4.set_ylabel('Eval g2op_end')
    ax4.set_title('Performance vs Training Time')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/trial_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/trial_performance.png")
    plt.close()


def plot_hyperparameter_analysis(df: pd.DataFrame, output_dir: str):
    """Analyze the impact of different hyperparameters."""

    # Filter out trials with NaN values in key hyperparameters
    hp_cols = ['lr', 'gamma', 'lambda', 'clip_param', 'entropy_coeff', 'hidden_dim']
    df_hp = df.dropna(subset=hp_cols)

    if len(df_hp) < 3:
        print("Not enough trials with complete hyperparameter data for analysis")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    hyperparams = ['lr', 'gamma', 'lambda', 'clip_param', 'entropy_coeff', 'hidden_dim']

    for idx, hp in enumerate(hyperparams):
        if hp in df_hp.columns:
            ax = axes[idx]
            scatter = ax.scatter(df_hp[hp], df_hp['eval_g2op_end'],
                               c=df_hp['eval_reward'], cmap='RdYlGn',
                               alpha=0.7, s=150, edgecolors='black', linewidth=0.5)
            ax.set_xlabel(hp)
            ax.set_ylabel('Eval g2op_end')
            ax.set_title(f'Impact of {hp}')
            if hp == 'lr':
                ax.set_xscale('log')
            plt.colorbar(scatter, ax=ax, label='Eval Reward')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/hyperparameter_impact.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/hyperparameter_impact.png")
    plt.close()


def plot_learning_curves(results_dir: str, output_dir: str, top_n: int = 5):
    """Plot learning curves for top N trials."""

    results_path = Path(results_dir)
    trial_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('CustomPPO_2748683_')]

    # Load and rank trials
    trial_scores = []
    for trial_dir in trial_dirs:
        progress_file = trial_dir / 'progress.csv'
        if progress_file.exists():
            try:
                df = pd.read_csv(progress_file)
                if not df.empty and 'evaluation/custom_metrics/g2op_end_timestep_mean' in df.columns:
                    max_score = df['evaluation/custom_metrics/g2op_end_timestep_mean'].max()
                    trial_scores.append((trial_dir, max_score))
            except:
                pass

    # Sort and get top N
    trial_scores.sort(key=lambda x: x[1], reverse=True)
    top_trials = trial_scores[:top_n]

    # Plot learning curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for trial_dir, score in top_trials:
        trial_id = trial_dir.name.split('_')[2]
        progress_file = trial_dir / 'progress.csv'
        df = pd.read_csv(progress_file)

        # Eval g2op_end
        if 'evaluation/custom_metrics/g2op_end_timestep_mean' in df.columns:
            axes[0, 0].plot(df['timesteps_total'],
                           df['evaluation/custom_metrics/g2op_end_timestep_mean'],
                           label=f'{trial_id} (max: {score:.0f})', alpha=0.7, linewidth=2)

        # Eval reward
        if 'evaluation/custom_metrics/episode_reward_mean' in df.columns:
            axes[0, 1].plot(df['timesteps_total'],
                           df['evaluation/custom_metrics/episode_reward_mean'],
                           label=trial_id, alpha=0.7, linewidth=2)

        # Train g2op_end
        if 'sampler_results/custom_metrics/g2op_end_timestep_mean' in df.columns:
            axes[1, 0].plot(df['timesteps_total'],
                           df['sampler_results/custom_metrics/g2op_end_timestep_mean'],
                           label=trial_id, alpha=0.7, linewidth=2)

        # Train reward
        if 'sampler_results/episode_reward_mean' in df.columns:
            axes[1, 1].plot(df['timesteps_total'],
                           df['sampler_results/episode_reward_mean'],
                           label=trial_id, alpha=0.7, linewidth=2)

    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Eval g2op_end')
    axes[0, 0].set_title(f'Top {top_n} Trials - Evaluation Performance')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('Eval Reward')
    axes[0, 1].set_title('Evaluation Reward')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Timesteps')
    axes[1, 0].set_ylabel('Train g2op_end')
    axes[1, 0].set_title('Training Performance')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].set_ylabel('Train Reward')
    axes[1, 1].set_title('Training Reward')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/learning_curves.png")
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate a text summary report."""

    report = []
    report.append("=" * 80)
    report.append("OPTUNA HYPERPARAMETER OPTIMIZATION SUMMARY")
    report.append("=" * 80)
    report.append(f"\nTotal Trials: {len(df)}")
    report.append(f"Completed Trials: {len(df[df['iterations'] > 0])}")
    report.append("\n" + "-" * 80)
    report.append("TOP 10 TRIALS BY EVALUATION PERFORMANCE")
    report.append("-" * 80)

    top10 = df.head(10)
    for idx, row in top10.iterrows():
        report.append(f"\nRank {len(report) - 6}:")
        report.append(f"  Trial ID: {row['trial_id']}")
        report.append(f"  Eval g2op_end: {row['eval_g2op_end']:.0f} timesteps")
        report.append(f"  Eval Reward: {row['eval_reward']:.2f}")
        report.append(f"  Train g2op_end: {row['train_g2op_end']:.0f} timesteps")
        report.append(f"  Total Iterations: {row['iterations']}")
        report.append(f"  Total Timesteps: {row['timesteps_total']:.0f}")

        if not pd.isna(row.get('lr')):
            report.append(f"  Hyperparameters:")
            report.append(f"    Learning Rate: {row['lr']:.6f}")
            report.append(f"    Gamma: {row.get('gamma', 'N/A')}")
            report.append(f"    Lambda: {row.get('lambda', 'N/A')}")
            report.append(f"    Clip Param: {row.get('clip_param', 'N/A')}")
            report.append(f"    Entropy Coeff: {row.get('entropy_coeff', 'N/A')}")
            report.append(f"    Hidden Dim: {row.get('hidden_dim', 'N/A')}")
            report.append(f"    Num Layers: {row.get('num_layers', 'N/A')}")

    report.append("\n" + "-" * 80)
    report.append("BEST HYPERPARAMETER VALUES")
    report.append("-" * 80)

    best_trial = df.iloc[0]
    report.append(f"\nBest Trial ID: {best_trial['trial_id']}")
    report.append(f"Best Eval Score: {best_trial['eval_g2op_end']:.0f} timesteps")

    if not pd.isna(best_trial.get('lr')):
        report.append(f"\nBest Hyperparameters:")
        for hp in ['lr', 'gamma', 'lambda', 'clip_param', 'entropy_coeff', 'vf_loss_coeff',
                   'hidden_dim', 'num_layers', 'aggregation']:
            if hp in best_trial and not pd.isna(best_trial[hp]):
                report.append(f"  {hp}: {best_trial[hp]}")

    report.append("\n" + "=" * 80)

    # Save report
    report_text = "\n".join(report)
    with open(f'{output_dir}/optimization_summary.txt', 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved: {output_dir}/optimization_summary.txt")

    # Save CSV
    df.to_csv(f'{output_dir}/all_trials_results.csv', index=False)
    print(f"Saved: {output_dir}/all_trials_results.csv")


def main():
    parser = argparse.ArgumentParser(description='Analyze Optuna hyperparameter optimization results')
    parser.add_argument('--results_dir', type=str,
                       default='/pfs/data6/home/ka/ka_iai/ka_hw6998/dev/NRI-for-explainable-RL-in-Power-Grids/results/experiments/gnn_hyperparameter_optimization',
                       help='Path to results directory')
    parser.add_argument('--output_dir', type=str,
                       default='./optuna_analysis',
                       help='Output directory for plots and reports')
    parser.add_argument('--top_n', type=int, default=5,
                       help='Number of top trials to show in learning curves')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading trial results...")
    df, metadata = load_trial_results(args.results_dir)

    if len(df) == 0:
        print("No trial results found!")
        return

    print(f"\nAnalyzing {len(df)} trials...")

    # Generate visualizations
    print("\nGenerating performance plots...")
    plot_trial_performance(df, args.output_dir)

    print("\nGenerating hyperparameter analysis...")
    plot_hyperparameter_analysis(df, args.output_dir)

    print("\nGenerating learning curves...")
    plot_learning_curves(args.results_dir, args.output_dir, args.top_n)

    print("\nGenerating summary report...")
    generate_summary_report(df, args.output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - trial_performance.png")
    print(f"  - hyperparameter_impact.png")
    print(f"  - learning_curves.png")
    print(f"  - optimization_summary.txt")
    print(f"  - all_trials_results.csv")


if __name__ == '__main__':
    main()

