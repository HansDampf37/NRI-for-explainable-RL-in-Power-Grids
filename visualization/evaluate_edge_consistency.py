"""
Evaluate and visualize consistency of edge-type probability predictions across multiple NRI training sessions.

Usage (from repo root):
    python visualization/evaluate_edge_consistency.py \
        --data-dir data/edge_probabilities \
        --prefix edges_testing_NRI_with_forecast \
        --out-dir output/edge_consistency \
        --top-k 20

This script assumes the .npy files with the provided prefix were stacked with session as first axis:
stacked shape: (n_sessions, n_edges, n_edge_types)

It computes per-edge mean, std, coefficient of variation (CV), entropy, pairwise Jensen-Shannon divergence across sessions,
argmax-agreement fraction and Fleiss' kappa. It writes PNG/PNG plots to the output directory.

If the stacked arrays have a different shape, the script will attempt to infer semantics and give helpful errors.

"""

import argparse
import os
from pathlib import Path
import glob
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import json
import pandas as pd
from typing import List, Tuple, Iterable, Sequence, TypedDict, Union, Dict, Optional
import numpy.typing as npt
from matplotlib.figure import Figure


class EdgeStats(TypedDict):
    mean_probs: npt.NDArray[np.floating]
    std_probs: npt.NDArray[np.floating]
    cv: npt.NDArray[np.floating]
    ent_mean: npt.NDArray[np.floating]
    ent_within: npt.NDArray[np.floating]
    agreement_fraction: npt.NDArray[np.floating]
    argmax_mode: npt.NDArray[np.int_]
    kappa_overall: float
    js_per_edge: npt.NDArray[np.floating]
    max_exists_probs: npt.NDArray[np.floating]
    mean_exists_probs: npt.NDArray[np.floating]


class FiguresDict(TypedDict):
    hist_js: Figure
    agreement_hist: Figure
    mean_existence_vs_variability: Figure
    max_existence_vs_variability: Figure
    session_js_heatmap: Figure
    # New combined 2x2 summary figure composing the four plots above
    summary: Figure
    edge_bars: Dict[int, Figure]


def load_and_stack(dir_path: str, prefix: str) -> Tuple[np.ndarray, List[str]]:
    """Load multiple .npy files sharing a prefix and stack them along the session axis.

    Assumption: Each file contains, per edge, a probability distribution over K edge types
    (typically shape (E, K)). Files are sorted alphabetically and stacked with a new
    first axis so that the resulting shape is (S, E, K), where S is the number of
    sessions/files.

    Args:
    - dir_path: Directory to search for files.
    - prefix: File prefix; all files "{prefix}*.npy" will be loaded.

    Returns:
    - stacked: np.ndarray with shape (S, ...), typically (S, E, K).
    - files: List of file paths in the order used (alphabetical).

    Raises:
    - FileNotFoundError: If no matching files are found.
    - ValueError: If the individual arrays have different shapes.
    """
    files = sorted(glob.glob(os.path.join(dir_path, f'{prefix}*.npy')))
    if not files:
        raise FileNotFoundError(f'No files with prefix {prefix} found in {dir_path}.')
    arrays = [np.load(f) for f in files]
    shapes = {a.shape for a in arrays}
    if len(shapes) > 1:
        raise ValueError(f'Inconsistent shapes for {prefix}: {shapes}')
    stacked = np.stack(arrays, axis=0)
    return stacked, files


def safe_cv(mean: Union[npt.NDArray[np.floating], float], std: Union[npt.NDArray[np.floating], float]) -> Union[npt.NDArray[np.floating], float]:
    """Compute the coefficient of variation (CV) elementwise as std / mean.

    Stabilization: If mean == 0, returns NaN to avoid division by zero and
    unstable values. NumPy broadcasting between mean and std is supported
    as long as they are broadcast-compatible.

    Args:
    - mean: np.ndarray or scalar; mean value(s) per entry.
    - std: np.ndarray or scalar; standard deviation(s) per entry.

    Returns:
    - cv: np.ndarray; same (broadcasted) shape as the inputs, with NaN where mean == 0.
    """
    # coefficient of variation, handle small means ()
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.where(mean == 0, np.nan, std / mean)
    return cv


def pairwise_js_between_sessions(stacked: npt.NDArray[np.floating], prior: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
    """Compute the mean pairwise Jensen–Shannon divergence between sessions.

    For every session pair (i, j) and for every edge e, compute the JS divergence
    between the type-probability distributions p_{i,e} and q_{j,e} (over K types).
    Before computing, each distribution is normalized to sum to 1; if the sum is 0,
    it is smoothed to a uniform distribution. The edge-wise JS values are then
    averaged across edges.

    Args:
    - stacked: np.ndarray of shape (S, E, K); S sessions, E edges, K edge types.
    - prior:

    Returns:
    - mat: np.ndarray of shape (S, S); symmetric matrix of mean JS divergences
      per session pair, base 2. Diagonal is 0.
    """
    # stacked: (S, E, K)
    _, E, K = stacked.shape
    if prior is not None:
        prior_baseline = np.tile(prior, (1, E, 1))
        stacked = np.concatenate([stacked, prior_baseline], axis=0)
    S, _, _ = stacked.shape

    mat = np.zeros((S, S))
    for i, j in itertools.combinations(range(S), 2):
        # compute average JS across edges
        vals = []
        for e in range(E):
            p = stacked[i, e]
            q = stacked[j, e]
            # ensure valid probability distributions
            p = np.asarray(p, dtype=float)
            q = np.asarray(q, dtype=float)
            # smoothing if zeros across all categories
            if p.sum() == 0:
                p = np.ones_like(p) / len(p)
            else:
                p = p / p.sum()
            if q.sum() == 0:
                q = np.ones_like(q) / len(q)
            else:
                q = q / q.sum()
            js = jensenshannon(p, q, base=2.0)
            vals.append(js)
        mean_js = float(np.nanmean(vals))
        mat[i, j] = mean_js
        mat[j, i] = mean_js
    return mat


def fleiss_kappa(annotations: npt.NDArray[np.int_], n_categories: int) -> float:
    """Compute Fleiss' kappa for discrete ratings from multiple raters.

    Assumption: Each edge is an "item" and each session is a "rater" that assigns exactly
    one category (argmax type) from {0, ..., K-1}. The function aggregates counts per item
    and category and applies the standard Fleiss' kappa formula.

    Args:
    - annotations: np.ndarray with shape (n_items, n_raters); integer labels in [0, n_categories).
    - n_categories: int; number of categories K.

    Returns:
    - kappa: float; Fleiss' kappa roughly in [-1, 1]. 0 ≈ chance, >0 above chance,
      <0 below chance. Returns NaN when not defined (e.g., <2 raters).
    """
    # annotations: array of shape (n_items, n_raters) with integer category labels in [0, n_categories)
    # returns Fleiss' kappa
    n_items, n_raters = annotations.shape
    if n_items <= 0:
        return np.nan
    if n_raters < 2:
        # kappa is not defined for fewer than 2 raters
        return np.nan
    # build n_ik: n_items x n_categories counts
    n_ik = np.zeros((n_items, n_categories), dtype=int)
    for i in range(n_items):
        for k in annotations[i]:
            n_ik[i, int(k)] += 1
    p_k = n_ik.sum(axis=0) / (n_items * n_raters)
    P_i = ( (n_ik * n_ik).sum(axis=1) - n_raters ) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = (p_k * p_k).sum()
    if (1 - P_e) == 0:
        return np.nan
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def compute_edge_statistics(stacked: npt.NDArray[np.floating]) -> EdgeStats:
    """Compute key statistics of edge classification distributions across sessions.

    Steps:
    1) Normalization: For each session s and edge e, normalize the length-K vector to sum to 1;
       if the sum is 0, use a uniform distribution.
    2) Mean and spread: mean_probs(E,K), std_probs(E,K) across sessions.
    3) Coefficient of variation: cv(E,K) = std/mean, with NaN where mean==0.
    4) Entropy: ent_mean(E,) = entropy of the mean distribution per edge; ent_within(E,) =
       mean entropy within sessions (first entropy per session/edge, then average over S).
    5) Argmax agreement: For each edge, the mode of argmax types; agreement_fraction(E,) = share
       of sessions in the mode.
    6) Fleiss' kappa: A single overall value across all edges (items) and sessions (raters)
       based on argmax labels.
    7) JS per edge: js_per_edge(E,) = mean pairwise Jensen–Shannon divergence across sessions.
    8) Maximum existing prob: Maximum (over sessions) existence probability.

    Args:
    - stacked: np.ndarray of shape (S, E, K); raw probabilities per session/edge/type.

    Returns (dict with the following entries):
    - 'mean_probs': np.ndarray (E, K)
    - 'std_probs': np.ndarray (E, K)
    - 'cv': np.ndarray (E, K)
    - 'ent_mean': np.ndarray (E,)
    - 'ent_within': np.ndarray (E,)
    - 'agreement_fraction': np.ndarray (E,)
    - 'argmax_mode': np.ndarray (E,) integer
    - 'kappa_overall': float
    - 'js_per_edge': np.ndarray (E,)
    - 'max_exists_prob': np.ndarray (E,)
    """
    # stacked: (S, E, K)
    S, E, K = stacked.shape
    # normalize each session distribution per edge
    stacked_norm = np.copy(stacked).astype(float)
    for s in range(S):
        for e in range(E):
            v = stacked_norm[s, e]
            ssum = v.sum()
            if ssum == 0:
                stacked_norm[s, e] = np.ones_like(v) / len(v)
            else:
                stacked_norm[s, e] = v / ssum
    mean_probs = stacked_norm.mean(axis=0)  # (E, K)
    std_probs = stacked_norm.std(axis=0)
    cv = safe_cv(mean_probs, std_probs)
    # entropy of mean distribution (per edge over types K)
    ent_mean = entropy(mean_probs, base=2.0, axis=1)  # (E,)
    # mean entropy within sessions: entropy over types (axis=2), then mean over sessions (axis=0)
    ent_within = np.asarray(entropy(stacked_norm, base=2.0, axis=2)).mean(axis=0)  # (E,)
    # argmax per session
    argmax = stacked_norm.argmax(axis=2)  # (S, E)
    # compute per-edge mode and counts with numpy to avoid SciPy version issues
    # argmax: shape (S, E) -> for each edge e, collect S integer labels in [0..K-1]
    argmax_mode = np.zeros(E, dtype=int)
    argmax_counts = np.zeros(E, dtype=int)
    for e in range(E):
        vals = argmax[:, e]
        # bincount up to K categories
        counts = np.bincount(vals, minlength=K)
        argmax_mode[e] = int(np.argmax(counts))
        argmax_counts[e] = int(np.max(counts))
    agreement_fraction = argmax_counts / float(S)
    # fleiss kappa across sessions for each edge? Fleiss is typically for multiple items across raters
    # we'll compute a single Fleiss' kappa across all edges treating each edge as an item and sessions as raters
    annotations = argmax.T  # (E, S) -> items x raters
    kappa_overall = fleiss_kappa(annotations, n_categories=K)
    # pairwise JS per edge
    # for each edge, compute mean pairwise JS across sessions
    from itertools import combinations
    js_per_edge = np.zeros(E)
    for e in range(E):
        pairs = []
        for i, j in combinations(range(S), 2):
            p = stacked_norm[i, e]
            q = stacked_norm[j, e]
            js = jensenshannon(p, q, base=2.0)
            pairs.append(js)
        js_per_edge[e] = float(np.nanmean(pairs)) if pairs else 0.0

    exists_prob = stacked[..., 1:].sum(axis=-1)
    max_exists_prob = exists_prob.max(0)
    mean_exists_prob = exists_prob.mean(0)
    stats = {
        'mean_probs': mean_probs,
        'std_probs': std_probs,
        'cv': cv,
        'ent_mean': ent_mean,
        'ent_within': ent_within,
        'agreement_fraction': agreement_fraction,
        'argmax_mode': argmax_mode,
        'kappa_overall': kappa_overall,
        'js_per_edge': js_per_edge,
        'max_exists_probs': max_exists_prob,
        'mean_exists_probs': mean_exists_prob,
    }
    return stats


def plot_hist_js(js_per_edge: npt.NDArray[np.floating], out_path: str) -> Figure:
    """Save a histogram of the per-edge mean JS divergence across sessions and return the figure.

    Args:
    - js_per_edge: np.ndarray (E,); mean pairwise JS divergence per edge.
    - out_path: Output file path (PNG).

    Returns: The created matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(js_per_edge, bins=50, ax=ax)
    ax.set_xlabel('Mean pairwise Jensen-Shannon (per edge)')
    ax.set_title('Distribution of per-edge JS divergence across sessions')
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_agreement_hist(agreement_fraction: npt.NDArray[np.floating], out_path: str) -> Figure:
    """Save a histogram of the argmax agreement fraction per edge and return the figure.

    Args:
    - agreement_fraction: np.ndarray (E,); fraction of sessions choosing the same argmax type.
    - out_path: Output file path (PNG).

    Returns: The created matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(agreement_fraction, bins=50, ax=ax)
    ax.set_xlabel('Argmax agreement fraction across sessions')
    ax.set_title('Agreement fraction distribution (argmax)')
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_max_existence_cv_scatter(max_exists_probs: npt.NDArray[np.floating], cv: npt.NDArray[np.floating],
                                  out_path: str, c: Optional[npt.NDArray[np.floating]] = None, c_label: Optional[str] = None) -> Figure:
    """Scatter plot of top existence probability vs. mean CV per edge and return the figure.

    For each edge, plot the highest existence probability (over S) against the mean CV (over K).

    Args:
    - max_exists_probs: np.ndarray (E, ); accumulate type probs except first and select maximum over sessions.
    - cv: np.ndarray (E, K); coefficient of variation per type and edge.
    - c: np.ndarray (E,); value to use for coloring
    - out_path: Output file path (PNG).

    Returns: The created matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(max_exists_probs, np.nanmean(cv, axis=1), c=c, cmap='viridis', s=20)
    if c_label is not None:
        fig.colorbar(sc, label=c_label)
    ax.set_xlabel('Max existence prob (over sessions) (per edge)')
    ax.set_ylabel('Mean (over sessions) CV across types (per edge)')
    ax.set_title('Existence probability vs variability (CV)')
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_mean_existence_cv_scatter(mean_exists_probs: npt.NDArray[np.floating], cv: npt.NDArray[np.floating],
                                  out_path: str, c: Optional[npt.NDArray[np.floating]] = None, c_label: Optional[str] = None) -> Figure:
    """Scatter plot of mean existence probability vs. mean CV per edge and return the figure.

    For each edge, plot the mean existence probability (over S) against the mean CV (over K).

    Args:
    - mean_exists_probs: np.ndarray (E, ); accumulate type probs except first and mean over sessions.
    - cv: np.ndarray (E, K); coefficient of variation per type and edge.
    - c: np.ndarray (E,); value to use for coloring
    - out_path: Output file path (PNG).

    Returns: The created matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(mean_exists_probs, np.nanmean(cv, axis=1), c=c, cmap='viridis', s=20)
    if c_label is not None:
        fig.colorbar(sc, label=c_label)
    ax.set_xlabel('Mean existence prob (over sessions) (per edge)')
    ax.set_ylabel('Mean (over sessions) CV across types (per edge)')
    ax.set_title('Existence probability vs variability (CV)')
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_session_js_heatmap(pairwise_js: npt.NDArray[np.floating], out_path: str) -> Figure:
    """Heatmap of the mean pairwise JS divergence between sessions and return the figure.

    Args:
    - pairwise_js: np.ndarray (S, S); entries are JS divergences averaged over edges.
    - out_path: Output file path (PNG).

    Returns: The created matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pairwise_js, cmap='magma', annot=True, fmt='.3f', ax=ax, vmin=0, vmax=1)
    ax.set_title('Mean pairwise JS between sessions (averaged over edges)')
    ax.set_xlabel('session')
    ax.set_ylabel('session')
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_edge_bar_for_examples(stacked: npt.NDArray[np.floating], edge_indices: Iterable[int], out_dir: str, files: Sequence[str]) -> Dict[int, Figure]:
    """Create bar charts of type probabilities per session for selected edges and return figures.

    Args:
    - stacked: np.ndarray (S, E, K); raw or already normalized probabilities.
    - edge_indices: iterable of int; edge indices to plot.
    - out_dir: Target directory for the PNG files.
    - files: List of source file names/paths; used for session labels.

    Returns: Dict mapping edge index to the created matplotlib Figure; files are also saved.
    """
    S, E, K = stacked.shape
    figs: Dict[int, Figure] = {}
    labels = [Path(f).stem for f in files]
    x = np.arange(S)
    for e in edge_indices:
        fig, ax = plt.subplots(figsize=(10, 4))
        arr = stacked[:, e, :]  # S x K
        width = 0.8 / K
        for k in range(K):
            ax.bar(x + k * width, arr[:, k], width=width, label=f'type{k}')
        ax.set_xticks(x + width * (K - 1) / 2)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Probability')
        ax.set_title(f'Edge {e} - per-session type probabilities')
        ax.legend()
        out_path = os.path.join(out_dir, f'edge_{e}_per_session_probs.png')
        fig.savefig(out_path)
        figs[int(e)] = fig
    return figs


def _create_summary_figure_from_pngs(out_dir: str) -> Figure:
    """Create a 2x2 summary figure by loading the already-saved PNG plots and composing them.

    This avoids duplicating plotting logic and works headless. The figure is saved as
    "summary.png" in out_dir and returned.
    """
    # Expected file names as saved by the individual plotting functions
    paths = {
        'hist_js': os.path.join(out_dir, 'hist_js_per_edge.png'),
        'agreement_hist': os.path.join(out_dir, 'agreement_fraction_hist.png'),
        'mean_existence_vs_variability': os.path.join(out_dir, 'mean_existence_vs_variability.png'),
        'max_existence_vs_variability': os.path.join(out_dir, 'max_existence_vs_variability.png'),
    }
    # Create composite figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    axes = axes.ravel()
    for ax, key in zip(axes, paths.keys()):
        try:
            img = plt.imread(paths[key])
            ax.imshow(img)
            ax.set_axis_off()
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Missing: {os.path.basename(paths[key])}', ha='center', va='center')
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'summary.png'))
    return fig


def run_analysis(data_dir: str, prefix: str, out_dir: str, top_k: int = 20) -> Tuple[EdgeStats, npt.NDArray[np.floating]]:
    """Run the complete analysis pipeline and write plots/tables.

    Steps:
    - Load and stack .npy files into (S, E, K).
    - Compute per-edge statistics and the global Fleiss' kappa.
    - Compute JS similarity between sessions.
    - Generate and save plots as well as CSV/JSON summaries.

    Args:
    - data_dir: Directory containing the input files.
    - prefix: File prefix to select ("{prefix}*.npy").
    - out_dir: Output directory; will be created if necessary.
    - top_k: Number of top edges (by mean and by JS) from which some example plots are produced.

    Returns:
    - stats: Dict of arrays/metrics computed by compute_edge_statistics().
    - pairwise_js: np.ndarray (S, S) of JS divergences between sessions.
    """
    stacked, files = load_and_stack(data_dir, prefix)
    # Expect stacked shape (S, E, K) but if files each contain (E, K) we have stacked as (S, E, K)
    if stacked.ndim != 3:
        raise ValueError(f'Expected stacked ndarray with ndim==3 (S,E,K), got shape {stacked.shape}')
    S, E, K = stacked.shape
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stats = compute_edge_statistics(stacked)
    # pairwise session JS
    pairwise_js = pairwise_js_between_sessions(stacked)
    # plots
    fig1 = plot_hist_js(stats['js_per_edge'], os.path.join(out_dir, 'hist_js_per_edge.png'))
    fig2 = plot_mean_existence_cv_scatter(stats['mean_exists_probs'], stats['cv'], os.path.join(out_dir, 'mean_existence_vs_variability.png'), stats['js_per_edge'], "Jensen-Shannon divergence")
    fig2_5 = plot_max_existence_cv_scatter(stats['max_exists_probs'], stats['cv'], os.path.join(out_dir, 'max_existence_vs_variability.png'), stats['js_per_edge'], "Jensen-Shannon divergence")
    fig3 = plot_session_js_heatmap(pairwise_js, os.path.join(out_dir, 'session_js_heatmap.png'))
    fig4 = plot_agreement_hist(stats['agreement_fraction'], os.path.join(out_dir, 'agreement_fraction_hist.png'))
    # choose edges to inspect: highest mean for any type, and highest js
    top_edges_by_mean = np.argsort(-stats['mean_probs'].max(axis=1))[:top_k]
    top_edges_by_js = np.argsort(-stats['js_per_edge'])[:top_k]
    example_edges = np.unique(np.concatenate([top_edges_by_mean[:5], top_edges_by_js[:5]]))
    figs_bars = plot_edge_bar_for_examples(stacked, example_edges, out_dir, files)
    # Close figures here to avoid leaking when run headless
    import matplotlib.pyplot as _plt
    for _f in (fig1, fig2, fig2_5, fig3, fig4):
        _plt.close(_f)
    for _f in figs_bars.values():
        _plt.close(_f)
    # save a small summary and per-edge CSV
    summary = {
        'n_sessions': int(S),
        'n_edges': int(E),
        'n_types': int(K),
        'kappa_overall': float(stats['kappa_overall']) if stats['kappa_overall'] is not None else None,
        'mean_js_over_edges': float(np.nanmean(stats['js_per_edge'])),
        'median_js_over_edges': float(np.nanmedian(stats['js_per_edge'])),
    }
    # save JSON summary
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    # per-edge CSV
    top_mean = stats['mean_probs'].max(axis=1)
    top_type = stats['mean_probs'].argmax(axis=1)
    mean_cv_per_edge = np.nanmean(stats['cv'], axis=1)
    df = pd.DataFrame({
        'edge_idx': np.arange(E),
        'top_mean_prob': top_mean,
        'top_type': top_type,
        'mean_entropy': stats['ent_mean'],
        'mean_within_entropy': stats['ent_within'],
        'mean_js': stats['js_per_edge'],
        'agreement_fraction': stats['agreement_fraction'],
        'mean_cv': mean_cv_per_edge,
    })
    df.to_csv(os.path.join(out_dir, 'per_edge_metrics.csv'), index=False)
    print('Saved results to', out_dir)
    return stats, pairwise_js


def run_analysis_with_figs(data_dir: str, prefix: str, out_dir: str, top_k: int = 20) -> Tuple[EdgeStats, npt.NDArray[np.floating], FiguresDict]:
    """Like run_analysis, but also returns the created matplotlib figures for inline display.

    Returns:
    - stats: same as run_analysis
    - pairwise_js: same as run_analysis
    - figs: dict with keys 'hist_js', 'mean_vs_cv', 'session_js_heatmap', 'agreement_hist', 'summary', and 'edge_bars' (dict edge->Figure)
    """
    stacked, files = load_and_stack(data_dir, prefix)
    if stacked.ndim != 3:
        raise ValueError(f'Expected stacked ndarray with ndim==3 (S,E,K), got shape {stacked.shape}')
    S, E, K = stacked.shape
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stats = compute_edge_statistics(stacked)
    pairwise_js = pairwise_js_between_sessions(stacked)
    figs: FiguresDict = {
        'hist_js': plot_hist_js(stats['js_per_edge'], os.path.join(out_dir, 'hist_js_per_edge.png')),
        'mean_existence_vs_variability': plot_mean_existence_cv_scatter(stats['mean_exists_probs'], stats['cv'], os.path.join(out_dir, 'mean_existence_vs_variability.png'), stats['js_per_edge'], "Jensen-Shannon divergence"),
        'max_existence_vs_variability': plot_max_existence_cv_scatter(stats['max_exists_probs'], stats['cv'], os.path.join(out_dir, 'max_existence_vs_variability.png'), stats['js_per_edge'], "Jensen-Shannon divergence"),
        'session_js_heatmap': plot_session_js_heatmap(pairwise_js, os.path.join(out_dir, 'session_js_heatmap.png')),
        'agreement_hist': plot_agreement_hist(stats['agreement_fraction'], os.path.join(out_dir, 'agreement_fraction_hist.png')),
        'summary': _create_summary_figure_from_pngs(out_dir),
        'edge_bars': {}
    }
    top_edges_by_mean = np.argsort(-stats['mean_probs'].max(axis=1))[:top_k]
    top_edges_by_js = np.argsort(-stats['js_per_edge'])[:top_k]
    example_edges = np.unique(np.concatenate([top_edges_by_mean[:5], top_edges_by_js[:5]]))
    figs['edge_bars'] = plot_edge_bar_for_examples(stacked, example_edges, out_dir, files)
    # Save summary and CSV (same as run_analysis)
    summary = {
        'n_sessions': int(S),
        'n_edges': int(E),
        'n_types': int(K),
        'kappa_overall': float(stats['kappa_overall']) if stats['kappa_overall'] is not None else None,
        'mean_js_over_edges': float(np.nanmean(stats['js_per_edge'])),
        'median_js_over_edges': float(np.nanmedian(stats['js_per_edge'])),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    top_mean = stats['mean_probs'].max(axis=1)
    top_type = stats['mean_probs'].argmax(axis=1)
    mean_cv_per_edge = np.nanmean(stats['cv'], axis=1)
    df = pd.DataFrame({
        'edge_idx': np.arange(E),
        'top_mean_prob': top_mean,
        'top_type': top_type,
        'mean_entropy': stats['ent_mean'],
        'mean_within_entropy': stats['ent_within'],
        'mean_js': stats['js_per_edge'],
        'agreement_fraction': stats['agreement_fraction'],
        'mean_cv': mean_cv_per_edge,
    })
    df.to_csv(os.path.join(out_dir, 'per_edge_metrics.csv'), index=False)
    print('Saved results to', out_dir)
    return stats, pairwise_js, figs


def main() -> None:
    """Command-line entry point.

    Parses arguments, calls run_analysis(), and exits. See the module docstring
    above for an example invocation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--out-dir', default='output/edge_consistency')
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args()
    run_analysis(args.data_dir, args.prefix, args.out_dir, args.top_k)


if __name__ == '__main__':
    main()
