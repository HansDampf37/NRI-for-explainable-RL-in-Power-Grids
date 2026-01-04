#!/usr/bin/env python3
"""
Create an interactive Optuna dashboard from Ray Tune results.
This script converts Ray Tune results to an Optuna study for visualization.
"""

import os
import json
import optuna
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List
from collections import defaultdict


def load_ray_tune_results(results_dir: str) -> List[Dict]:
    """Load Ray Tune trial results and convert to list of dicts."""
    results_path = Path(results_dir)
    trial_data = []

    trial_dirs = [d for d in results_path.iterdir()
                  if d.is_dir() and d.name.startswith('CustomPPO_2748683_')]

    print(f"Found {len(trial_dirs)} trials")

    for trial_dir in trial_dirs:
        trial_id = trial_dir.name.split('_')[2]

        # Load params and progress
        params_file = trial_dir / 'params.json'
        progress_file = trial_dir / 'progress.csv'

        if params_file.exists() and progress_file.exists():
            try:
                with open(params_file, 'r') as f:
                    params = json.load(f)

                df = pd.read_csv(progress_file)
                if not df.empty:
                    last_row = df.iloc[-1]

                    trial_info = {
                        'trial_id': trial_id,
                        'trial_name': trial_dir.name,
                        'params': params,
                        'score': last_row.get('evaluation/custom_metrics/g2op_end_timestep_mean', 0),
                        'eval_reward': last_row.get('evaluation/custom_metrics/episode_reward_mean', 0),
                        'timesteps': last_row.get('timesteps_total', 0),
                        'iterations': len(df),
                    }
                    trial_data.append(trial_info)
            except Exception as e:
                print(f"Warning: Could not load {trial_dir.name}: {e}")

    return trial_data


def create_optuna_study(trial_data: List[Dict], storage_path: str) -> optuna.Study:
    """Create an Optuna study from trial data."""

    # Create study with SQLite storage
    storage = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name="gnn_hyperparameter_optimization",
        storage=storage,
        direction="maximize",
        load_if_exists=False
    )

    print(f"\nCreating Optuna study with {len(trial_data)} trials...")

    # First pass: collect all parameter values to determine ranges
    param_values = defaultdict(list)
    for trial_info in trial_data:
        params = trial_info['params']
        for key in ['lr', 'gamma', 'lambda', 'clip_param', 'entropy_coeff',
                    'vf_loss_coeff', 'kl_coeff', 'train_batch_size',
                    'sgd_minibatch_size', 'num_sgd_iter']:
            if key in params and params[key] is not None:
                param_values[key].append(params[key])

    # Determine distributions based on actual values
    distributions = {}
    for key, values in param_values.items():
        if not values:
            continue

        min_val = min(values)
        max_val = max(values)

        # Add some padding to ranges
        if key in ['lr', 'entropy_coeff', 'kl_coeff', 'vf_loss_coeff']:
            # Float parameters
            if min_val > 0 and max_val / min_val > 10:
                # Log scale for parameters spanning orders of magnitude
                distributions[key] = optuna.distributions.FloatDistribution(
                    min_val * 0.5, max_val * 2.0, log=True
                )
            else:
                distributions[key] = optuna.distributions.FloatDistribution(
                    max(0, min_val * 0.9), max_val * 1.1, log=False
                )
        elif key in ['gamma', 'lambda', 'clip_param']:
            # Float parameters without log
            distributions[key] = optuna.distributions.FloatDistribution(
                max(0, min_val * 0.9), min(1.0, max_val * 1.1), log=False
            )
        else:
            # Integer parameters
            distributions[key] = optuna.distributions.IntDistribution(
                int(min_val * 0.8), int(max_val * 1.2)
            )

    # Second pass: add trials to study
    for trial_info in trial_data:
        params = trial_info['params']

        # Extract hyperparameters that exist in this trial
        trial_params = {}
        for key in distributions.keys():
            if key in params and params[key] is not None:
                trial_params[key] = params[key]

        # Create trial only if we have parameters
        if trial_params:
            try:
                trial = optuna.trial.create_trial(
                    params=trial_params,
                    distributions={k: distributions[k] for k in trial_params.keys()},
                    user_attrs={
                        'trial_id': trial_info['trial_id'],
                        'trial_name': trial_info['trial_name'],
                        'eval_reward': trial_info['eval_reward'],
                        'timesteps': trial_info['timesteps'],
                        'iterations': trial_info['iterations'],
                    },
                    value=trial_info['score'],
                )
                study.add_trial(trial)
            except Exception as e:
                print(f"Warning: Could not add trial {trial_info['trial_id']}: {e}")

    return study


def generate_optuna_visualizations(study: optuna.Study, output_dir: str):
    """Generate Optuna visualization plots."""
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
    )

    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating Optuna visualizations...")

    # Optimization history
    try:
        fig = plot_optimization_history(study)
        fig.savefig(f'{output_dir}/optuna_optimization_history.png', dpi=300, bbox_inches='tight')
        print(f"  - Saved optimization_history.png")
        plt.close()
    except Exception as e:
        print(f"  - Could not generate optimization history: {e}")

    # Parameter importances
    try:
        fig = plot_param_importances(study)
        fig.savefig(f'{output_dir}/optuna_param_importances.png', dpi=300, bbox_inches='tight')
        print(f"  - Saved param_importances.png")
        plt.close()
    except Exception as e:
        print(f"  - Could not generate param importances: {e}")

    # Parallel coordinate plot
    try:
        fig = plot_parallel_coordinate(study)
        fig.savefig(f'{output_dir}/optuna_parallel_coordinate.png', dpi=300, bbox_inches='tight')
        print(f"  - Saved parallel_coordinate.png")
        plt.close()
    except Exception as e:
        print(f"  - Could not generate parallel coordinate: {e}")

    # Slice plot
    try:
        fig = plot_slice(study)
        fig.savefig(f'{output_dir}/optuna_slice.png', dpi=300, bbox_inches='tight')
        print(f"  - Saved slice.png")
        plt.close()
    except Exception as e:
        print(f"  - Could not generate slice plot: {e}")


def print_study_summary(study: optuna.Study):
    """Print summary of the study."""
    print("\n" + "=" * 80)
    print("OPTUNA STUDY SUMMARY")
    print("=" * 80)

    print(f"\nTotal Trials: {len(study.trials)}")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value (g2op_end): {study.best_value:.0f} timesteps")

    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\nBest Trial Attributes:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Create Optuna dashboard from Ray Tune results')
    parser.add_argument('--results_dir', type=str,
                       default='/pfs/data6/home/ka/ka_iai/ka_hw6998/dev/NRI-for-explainable-RL-in-Power-Grids/results/experiments/gnn_hyperparameter_optimization',
                       help='Path to Ray Tune results directory')
    parser.add_argument('--output_dir', type=str,
                       default='./optuna_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--storage', type=str,
                       default='./optuna_analysis/optuna_study.db',
                       help='Path to SQLite database for Optuna study')

    args = parser.parse_args()

    # Load trial data
    print("Loading Ray Tune results...")
    trial_data = load_ray_tune_results(args.results_dir)

    if not trial_data:
        print("No trial data found!")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create Optuna study
    study = create_optuna_study(trial_data, args.storage)

    # Print summary
    print_study_summary(study)

    # Generate visualizations
    generate_optuna_visualizations(study, args.output_dir)

    print("\n" + "=" * 80)
    print("DASHBOARD CREATION COMPLETE!")
    print("=" * 80)
    print(f"\nOptuna study database: {args.storage}")
    print(f"Visualizations saved to: {args.output_dir}")
    print("\nTo launch the interactive Optuna dashboard, run:")
    print(f"  optuna-dashboard {args.storage}")
    print("\nThen open your browser to: http://localhost:8080")
    print("=" * 80)


if __name__ == '__main__':
    main()
