"""
Script to extract and visualize Optuna hyperparameter optimization results from Ray Tune experiments.
This script converts Ray Tune results to Optuna format for visualization with Optuna Dashboard.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import optuna
import pandas as pd
from ray.tune import ExperimentAnalysis


def extract_ray_tune_results(experiment_path: str) -> pd.DataFrame:
    """
    Extract results from Ray Tune experiment directory.

    Args:
        experiment_path: Path to the Ray Tune experiment directory

    Returns:
        DataFrame with trial results
    """
    print(f"Loading experiment from: {experiment_path}")

    # Try to load using ExperimentAnalysis
    try:
        analysis = ExperimentAnalysis(experiment_path)
        df = analysis.dataframe()
        print(f"Successfully loaded {len(df)} trials from Ray Tune")
        return df
    except Exception as e:
        print(f"Could not load with ExperimentAnalysis: {e}")
        print("Trying manual extraction...")

    # Manual extraction from individual trial directories
    trials = []
    exp_dir = Path(experiment_path)

    for trial_dir in exp_dir.iterdir():
        if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
            continue

        # Look for params.pkl and result files
        params_file = trial_dir / "params.pkl"
        result_file = trial_dir / "result.json"

        trial_data = {}

        # Load parameters
        if params_file.exists():
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
                trial_data.update(params)

        # Load results
        if result_file.exists():
            with open(result_file, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    # Keep only the last result (final metrics)
                trial_data.update(result)

        if trial_data:
            trial_data['trial_dir'] = str(trial_dir)
            trials.append(trial_data)

    if trials:
        df = pd.DataFrame(trials)
        print(f"Manually extracted {len(df)} trials")
        return df
    else:
        print("No trials found!")
        return pd.DataFrame()


def create_optuna_study_from_raytune(
    experiment_path: str,
    study_name: str,
    storage: Optional[str] = None,
    metric_name: str = "evaluation/custom_metrics/grid2op_end_mean",
    maximize: bool = True
) -> optuna.Study:
    """
    Create an Optuna study from Ray Tune results.

    Args:
        experiment_path: Path to Ray Tune experiment
        study_name: Name for the Optuna study
        storage: Optuna storage URL (e.g., 'sqlite:///optuna.db')
        metric_name: Name of the metric to optimize
        maximize: Whether to maximize the metric

    Returns:
        Optuna study object
    """
    # Extract Ray Tune results
    df = extract_ray_tune_results(experiment_path)

    if df.empty:
        raise ValueError("No trials found in experiment directory")

    # Create Optuna study
    direction = "maximize" if maximize else "minimize"

    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction
        )

    # Convert Ray Tune trials to Optuna trials
    print(f"\nConverting {len(df)} trials to Optuna format...")
    print(f"Target metric: {metric_name}")

    # Identify hyperparameters (config columns)
    config_cols = [col for col in df.columns if col.startswith('config/')]

    for idx, row in df.iterrows():
        # Extract hyperparameters
        params = {}
        for col in config_cols:
            param_name = col.replace('config/', '')
            value = row[col]
            if pd.notna(value):
                params[param_name] = value

        # Extract metric value
        metric_value = None
        if metric_name in df.columns:
            metric_value = row[metric_name]
        else:
            # Try to find the metric in nested structure
            for col in df.columns:
                if metric_name in col:
                    metric_value = row[col]
                    break

        if pd.isna(metric_value):
            print(f"Warning: Trial {idx} has no value for {metric_name}, skipping")
            continue

        # Create Optuna trial
        trial = optuna.trial.create_trial(
            params=params,
            distributions={
                name: optuna.distributions.CategoricalDistribution([value])
                for name, value in params.items()
            },
            values=[float(metric_value)],
        )

        study.add_trial(trial)

    print(f"Successfully created Optuna study with {len(study.trials)} trials")
    return study


def print_study_summary(study: optuna.Study):
    """Print a summary of the Optuna study."""
    print("\n" + "="*80)
    print("OPTUNA STUDY SUMMARY")
    print("="*80)

    print(f"\nStudy name: {study.study_name}")
    print(f"Direction: {study.direction.name}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    if study.best_trial:
        print(f"\nBest trial:")
        print(f"  Value: {study.best_value}")
        print(f"  Params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

    print("\n" + "="*80)


def visualize_with_optuna(study: optuna.Study, output_dir: str):
    """
    Create Optuna visualization plots.

    Args:
        study: Optuna study
        output_dir: Directory to save plots
    """
    import optuna.visualization as vis
    import plotly

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating Optuna visualizations in {output_dir}...")

    # Optimization history
    try:
        fig = vis.plot_optimization_history(study)
        plotly.offline.plot(fig, filename=os.path.join(output_dir, "optimization_history.html"))
        print("  ✓ optimization_history.html")
    except Exception as e:
        print(f"  ✗ Could not create optimization history: {e}")

    # Parameter importances
    try:
        fig = vis.plot_param_importances(study)
        plotly.offline.plot(fig, filename=os.path.join(output_dir, "param_importances.html"))
        print("  ✓ param_importances.html")
    except Exception as e:
        print(f"  ✗ Could not create parameter importances: {e}")

    # Parallel coordinate plot
    try:
        fig = vis.plot_parallel_coordinate(study)
        plotly.offline.plot(fig, filename=os.path.join(output_dir, "parallel_coordinate.html"))
        print("  ✓ parallel_coordinate.html")
    except Exception as e:
        print(f"  ✗ Could not create parallel coordinate plot: {e}")

    # Slice plot
    try:
        fig = vis.plot_slice(study)
        plotly.offline.plot(fig, filename=os.path.join(output_dir, "slice.html"))
        print("  ✓ slice.html")
    except Exception as e:
        print(f"  ✗ Could not create slice plot: {e}")

    # Contour plot (if there are multiple parameters)
    try:
        if len(study.best_params) >= 2:
            fig = vis.plot_contour(study)
            plotly.offline.plot(fig, filename=os.path.join(output_dir, "contour.html"))
            print("  ✓ contour.html")
    except Exception as e:
        print(f"  ✗ Could not create contour plot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Optuna hyperparameter optimization results from Ray Tune"
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Path to Ray Tune experiment directory"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="gnn_hyperparameter_optimization",
        help="Name for the Optuna study"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., 'sqlite:///optuna.db'). If not provided, in-memory study is used."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="evaluation/custom_metrics/grid2op_end_mean",
        help="Metric to optimize"
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        default=True,
        help="Maximize the metric (use --no-maximize to minimize)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./optuna_visualizations",
        help="Directory to save visualization plots"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )

    args = parser.parse_args()

    # Set default storage if not provided
    if args.storage is None:
        storage_dir = os.path.join(os.path.dirname(args.experiment_path), "optuna_storage")
        os.makedirs(storage_dir, exist_ok=True)
        args.storage = f"sqlite:///{os.path.join(storage_dir, f'{args.study_name}.db')}"
        print(f"Using default storage: {args.storage}")

    # Create Optuna study from Ray Tune results
    study = create_optuna_study_from_raytune(
        args.experiment_path,
        args.study_name,
        args.storage,
        args.metric,
        args.maximize
    )

    # Print summary
    print_study_summary(study)

    # Generate visualizations
    if not args.no_plots:
        visualize_with_optuna(study, args.output_dir)

    print(f"\n✓ Done! Optuna study saved to: {args.storage}")

    if args.storage and args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "")
        print(f"\nTo launch Optuna Dashboard, run:")
        print(f"  optuna-dashboard {db_path}")
        print(f"\nThen open your browser to http://localhost:8080")


if __name__ == "__main__":
    main()

