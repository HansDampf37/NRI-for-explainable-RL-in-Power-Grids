import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from numpy import dtype, ndarray
from scipy.stats import spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score, average_precision_score

from evaluate_rllib_agent import load_config, load_rllib_agent
from src.experiments.analyze_latent_graphs.MetricAnalyzer import PosteriorAnalyzer
from src.experiments.analyze_latent_graphs.agent_analysis_framework import LatentGraphAnalysisAgent
from src.rl4pnc.evaluation.evaluation_agents import RllibAgent
from src.rl4pnc.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment

logger = logging.getLogger(__name__)

class Hypothesis2verifier(PosteriorAnalyzer):
    """
    Computes per-timestep alignment metrics between posterior edge existence probabilities
    and risk coupling (edge-aligned, fully-connected ordering).

    Metrics per timestep:
      - Spearman rho
      - Pearson r
      - Kendall tau
      - Top-k overlap (default: top 10%)
      - ROC-AUC for predicting "strong Risk coupling" edges (threshold: top 10% of coupling)
      - Average Precision (AP) for same labeling (often more informative for imbalanced labels)
      - Mutual information MI(P -> C)

    Plus edge-wise temporal correlation (Pearson) over time:
      corr_e = corr_t(P_t,e, C_t,e) for each edge e (ignoring NaNs)
    """

    def __init__(
        self,
        out_dir: Path = Path("results/hypothesis_risk_coupling"),
        topk_frac: float = 0.05,
        strong_label_percentile: float = 90.0,
        min_samples_per_timestep: int = 10,
        min_timesteps_per_edge: int = 5,
        use_abs_coupling_for_metrics: bool = False,
        random_state_mi: int = 0,
        compare_to_mean_coupling: bool = False,
    ):
        self.topk_frac = float(topk_frac)
        self.strong_label_percentile = float(strong_label_percentile)
        self.min_samples_per_timestep = int(min_samples_per_timestep)
        self.min_timesteps_per_edge = int(min_timesteps_per_edge)
        self.use_abs_coupling_for_metrics = bool(use_abs_coupling_for_metrics)
        self.random_state_mi = int(random_state_mi)
        self.compare_to_mean_coupling = bool(compare_to_mean_coupling)
        self.mean_coupling = np.load(out_dir / "risk_coupling/coupling_mean.npy") if compare_to_mean_coupling else None
        # Results go into a mode-specific subdirectory so both modes can coexist
        self.outdir = out_dir / ("mean_coupling" if compare_to_mean_coupling else "risk_coupling")

        # Per-timestep scalars
        self.spearman_rho = []
        self.pearson_r = []
        self.kendall_tau = []
        self.topk_overlap = []
        self.roc_auc = []
        self.avg_precision = []
        self.mutual_info = []

        # For edge-wise temporal correlation
        self._P_over_time = []  # list of [E] with NaNs for invalid
        self._C_over_time = []  # list of [E] with NaNs for invalid

        # Optional: episode boundaries (if you want later)
        self._episode_ids = []
        self._t_global = 0

    @property
    def _coupling_label(self) -> str:
        """Human-readable label for what coupling is being compared against."""
        return "avg Risk coupling" if self.compare_to_mean_coupling else "Risk coupling"

    @property
    def _coupling_short(self) -> str:
        """Short token used in file names."""
        return "mean_risk_coupling" if self.compare_to_mean_coupling else "risk_coupling"

    @staticmethod
    def _nan_pearson(x: np.ndarray, y: np.ndarray) -> float:
        """Pearson with NaN filtering; returns np.nan if insufficient samples or zero variance."""
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        x0 = x[m]
        y0 = y[m]
        sx = x0.std()
        sy = y0.std()
        if sx < 1e-12 or sy < 1e-12:
            return np.nan
        return float(np.corrcoef(x0, y0)[0, 1])

    @staticmethod
    def _topk_overlap(c: np.ndarray, p: np.ndarray, frac: float) -> float:
        m = np.isfinite(c) & np.isfinite(p)
        c = c[m]
        p = p[m]
        n = c.shape[0]
        if n < 2:
            return np.nan
        k = max(1, int(frac * n))
        # indices of top-k
        top_c = np.argpartition(c, -k)[-k:]
        top_p = np.argpartition(p, -k)[-k:]
        overlap = np.intersect1d(top_c, top_p).size / k
        return float(overlap)

    @staticmethod
    def _auc_metrics(c: np.ndarray, p: np.ndarray, percentile: float):
        """
        Binarize coupling by percentile, compute ROC-AUC and AP. Returns (auc, ap) possibly NaN.
        """
        m = np.isfinite(c) & np.isfinite(p)
        c = c[m]
        p = p[m]
        if c.size < 10:
            return np.nan, np.nan

        thr = np.percentile(c, percentile)
        y = (c >= thr).astype(np.int32)
        # Need both classes present
        if y.min() == y.max():
            return np.nan, np.nan
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = np.nan
        try:
            ap = float(average_precision_score(y, p))
        except ValueError:
            ap = np.nan
        return auc, ap

    def _mi(self, c: np.ndarray, p: np.ndarray) -> float:
        """
        Mutual information between p (as feature) and c (target).
        Uses sklearn mutual_info_regression.
        """
        m = np.isfinite(c) & np.isfinite(p)
        c = c[m]
        p = p[m]
        if c.size < 10:
            return np.nan
        # MI expects X: (n_samples, n_features)
        X = p.reshape(-1, 1)
        try:
            mi = mutual_info_regression(X, c, random_state=self.random_state_mi)
            return float(mi[0])
        except Exception:
            return np.nan

    def on_rl_step(
        self,
        posterior: npt.NDArray,
        powergrid_graph: npt.NDArray,
        observation: BaseObservation,
        environment: Environment,
    ):
        # Coupling aligned with fully-connected edge ordering
        if self.compare_to_mean_coupling:
            coupling = self.mean_coupling
        else:
            coupling = get_risk_coupling_index(environment).astype(np.float64)  # [E]

        posterior_existence = posterior[:, 0].astype(np.float64) # [E]
        assert coupling.shape == posterior_existence.shape

        valid = np.isfinite(coupling) & np.isfinite(posterior_existence)

        # Optional: compare against |coupling|
        C = np.abs(coupling) if self.use_abs_coupling_for_metrics else coupling
        P = posterior_existence

        # Store full vectors (NaN for invalid) for edge-wise temporal correlation
        C_full = np.full_like(C, np.nan, dtype=np.float64)
        P_full = np.full_like(P, np.nan, dtype=np.float64)
        C_full[valid] = C[valid]
        P_full[valid] = P[valid]
        self._C_over_time.append(C_full)
        self._P_over_time.append(P_full)

        # Per-timestep metrics on valid subset
        C_valid = C[valid]
        P_valid = P[valid]

        if C_valid.size < self.min_samples_per_timestep:
            # Append NaNs to keep alignment with timesteps
            self.spearman_rho.append(np.nan)
            self.pearson_r.append(np.nan)
            self.kendall_tau.append(np.nan)
            self.topk_overlap.append(np.nan)
            self.roc_auc.append(np.nan)
            self.avg_precision.append(np.nan)
            self.mutual_info.append(np.nan)
            self._t_global += 1
            return

        # Spearman
        rho = spearmanr(C_valid, P_valid).correlation
        self.spearman_rho.append(float(rho) if rho is not None else np.nan)

        # Pearson
        r = self._nan_pearson(C_valid, P_valid)
        self.pearson_r.append(r)

        # Kendall
        tau = kendalltau(C_valid, P_valid).correlation
        self.kendall_tau.append(float(tau) if tau is not None else np.nan)

        # Top-k overlap
        self.topk_overlap.append(self._topk_overlap(C_valid, P_valid, self.topk_frac))

        # AUC + AP (strong coupling label by percentile)
        auc, ap = self._auc_metrics(C_valid, P_valid, self.strong_label_percentile)
        self.roc_auc.append(auc)
        self.avg_precision.append(ap)

        # Mutual information
        self.mutual_info.append(self._mi(C_valid, P_valid))

        self._t_global += 1

    def on_heuristic_step(self, powergrid_graph: npt.NDArray, observation: BaseObservation, environment: Environment):
        pass

    def on_new_episode(self, chronic_id: str):
        self._episode_ids.append(chronic_id)

    @staticmethod
    def _save_array(arr: np.ndarray, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    @staticmethod
    def _save_hist(values: np.ndarray, title: str, xlabel: str, outpath: Path,
                   vline: float | None = None, vline_label: str | None = None):
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        finite_values = values[np.isfinite(values)]
        sns.histplot(finite_values, kde=False)
        mean_val = float(np.mean(finite_values)) if finite_values.size > 0 else None
        if mean_val is not None:
            plt.axvline(mean_val, color="blue", linestyle="-.", label=f"Mean = {mean_val:.3f}")
        if vline is not None:
            plt.axvline(vline, color="red", linestyle="--",
                        label=vline_label if vline_label else f"x = {vline:.3f}")
        if vline is not None or mean_val is not None:
            plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(outpath.parent / (outpath.stem + ".png"))
        plt.savefig(outpath.parent / (outpath.stem + ".svg"))
        plt.show()

    @staticmethod
    def _save_scatter(values: np.ndarray, title: str, xlabel: str, ylabel: str, outpath: Path):
        outpath.parent.mkdir(parents=True, exist_ok=True)
        x, y = values
        mask = np.isfinite(x) & np.isfinite(y)
        plt.figure()
        plt.scatter(x=x[mask], y=y[mask], s=4, alpha=0.2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(outpath.parent / (outpath.stem + ".png"))
        plt.savefig(outpath.parent / (outpath.stem + ".svg"))
        plt.show()

    def on_evaluation_end(self):
        # Convert lists -> arrays
        spearman_rho = np.array(self.spearman_rho, dtype=np.float64)
        pearson_r = np.array(self.pearson_r, dtype=np.float64)
        kendall_tau_arr = np.array(self.kendall_tau, dtype=np.float64)
        topk_overlap_arr = np.array(self.topk_overlap, dtype=np.float64)
        roc_auc_arr = np.array(self.roc_auc, dtype=np.float64)
        ap_arr = np.array(self.avg_precision, dtype=np.float64)
        mi_arr = np.array(self.mutual_info, dtype=np.float64)

        num_timesteps = len(spearman_rho)

        # Edge-wise temporal correlation (Pearson across time, per edge)
        P_T = np.stack(self._P_over_time, axis=0)  # [T, E]
        C_T = np.stack(self._C_over_time, axis=0)  # [T, E]
        mean_c = np.nanmean(C_T, axis=0)  # [E]
        T, E = P_T.shape

        edge_corr = np.full(E, np.nan, dtype=np.float64)
        edge_spearman = np.full(E, np.nan, dtype=np.float64)
        for e in range(E):
            p_e = P_T[:, e]
            c_e = C_T[:, e]
            m = np.isfinite(p_e) & np.isfinite(c_e)
            if m.sum() < self.min_timesteps_per_edge:
                continue
            pe = p_e[m]
            ce = c_e[m]
            if pe.std() < 1e-12 or ce.std() < 1e-12:
                continue
            edge_corr[e] = np.corrcoef(pe, ce)[0, 1]
            rho_e = spearmanr(pe, ce).correlation
            edge_spearman[e] = float(rho_e) if rho_e is not None else np.nan

        # Print summaries
        def summarize(name: str, x: np.ndarray):
            xf = x[np.isfinite(x)]
            if xf.size == 0:
                print(f"{name}: all NaN")
                return
            print(f"{name} over t={num_timesteps}: mean={xf.mean():.4f}, std={xf.std():.4f}, median={np.median(xf):.4f}, n_valid={xf.size}")

        summarize("Spearman rho", spearman_rho)
        summarize("Pearson r", pearson_r)
        summarize("Kendall tau", kendall_tau_arr)
        summarize(f"Top-{int(self.topk_frac*100)}% overlap", topk_overlap_arr)
        summarize(f"ROC-AUC (labels: coupling >= p{self.strong_label_percentile})", roc_auc_arr)
        summarize(f"Avg Precision (labels: coupling >= p{self.strong_label_percentile})", ap_arr)
        summarize("Mutual information", mi_arr)

        edge_corr_f = edge_corr[np.isfinite(edge_corr)]
        if edge_corr_f.size:
            print(f"Edge-wise temporal Pearson corr over edges={E}: mean={edge_corr_f.mean():.4f}, std={edge_corr_f.std():.4f}, median={np.median(edge_corr_f):.4f}, n_valid={edge_corr_f.size}")
        else:
            print("Edge-wise temporal Pearson corr: all NaN")

        edge_spearman_f = edge_spearman[np.isfinite(edge_spearman)]
        if edge_spearman_f.size:
            print(rf"Edge-wise temporal Spearman rho over edges={E}: mean={edge_spearman_f.mean():.4f}, std={edge_spearman_f.std():.4f}, median={np.median(edge_spearman_f):.4f}, n_valid={edge_spearman_f.size}")
        else:
            print("Edge-wise temporal Spearman rho: all NaN")

        # Save arrays
        cs = self._coupling_short
        self._save_array(spearman_rho, self.outdir / f"spearman_{cs}_posterior.npy")
        self._save_array(pearson_r, self.outdir / f"pearson_{cs}_posterior.npy")
        self._save_array(kendall_tau_arr, self.outdir / f"kendall_{cs}_posterior.npy")
        self._save_array(topk_overlap_arr, self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct.npy")
        self._save_array(roc_auc_arr, self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}.npy")
        self._save_array(ap_arr, self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}.npy")
        self._save_array(mi_arr, self.outdir / f"mutual_info_{cs}_posterior.npy")
        self._save_array(edge_corr, self.outdir / f"edgewise_temporal_pearson_{cs}_posterior.npy")
        self._save_array(edge_spearman, self.outdir / f"edgewise_temporal_spearman_{cs}_posterior.npy")
        self._save_array(P_T, self.outdir / "posterior_over_time.npy")
        self._save_array(C_T, self.outdir / "coupling_over_time.npy")
        self._save_array(mean_c, self.outdir / "coupling_mean.npy")

        # Save plots (separate figure per metric)
        self.generate_plots(C_T, P_T, ap_arr, edge_corr, edge_spearman, kendall_tau_arr, mi_arr, pearson_r,
                            roc_auc_arr, spearman_rho, topk_overlap_arr)

    def repaint(self):
        """
        Reads all previously saved arrays from outdir and regenerates all plots
        by calling generate_plots with the loaded data.
        """
        cs = self._coupling_short
        spearman_rho = np.load(self.outdir / f"spearman_{cs}_posterior.npy")
        pearson_r = np.load(self.outdir / f"pearson_{cs}_posterior.npy")
        kendall_tau_arr = np.load(self.outdir / f"kendall_{cs}_posterior.npy")
        topk_overlap_arr = np.load(self.outdir / f"topk_overlap_{int(self.topk_frac*100)}pct.npy")
        roc_auc_arr = np.load(self.outdir / f"roc_auc_strong_{cs}_p{int(self.strong_label_percentile)}.npy")
        ap_arr = np.load(self.outdir / f"avg_precision_strong_{cs}_p{int(self.strong_label_percentile)}.npy")
        mi_arr = np.load(self.outdir / f"mutual_info_{cs}_posterior.npy")
        edge_corr = np.load(self.outdir / f"edgewise_temporal_pearson_{cs}_posterior.npy")
        edge_spearman = np.load(self.outdir / f"edgewise_temporal_spearman_{cs}_posterior.npy")
        P_T = np.load(self.outdir / "posterior_over_time.npy")
        C_T = np.load(self.outdir / "coupling_over_time.npy")

        self.generate_plots(C_T, P_T, ap_arr, edge_corr, edge_spearman, kendall_tau_arr, mi_arr, pearson_r,
                            roc_auc_arr, spearman_rho, topk_overlap_arr)

    def generate_plots(self, C_T: ndarray[Any, dtype[Any]], P_T: ndarray[Any, dtype[Any]],
                       ap_arr: ndarray[Any, dtype[Any]], edge_corr: ndarray[Any, dtype[Any]],
                       edge_spearman: ndarray[Any, dtype[Any]],
                       kendall_tau_arr: ndarray[Any, dtype[Any]], mi_arr: ndarray[Any, dtype[Any]],
                       pearson_r: ndarray[Any, dtype[Any]], roc_auc_arr: ndarray[Any, dtype[Any]],
                       spearman_rho: ndarray[Any, dtype[Any]], topk_overlap_arr: ndarray[Any, dtype[Any]]):
        sns.reset_orig()
        matplotlib.rcParams.update({
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
        })

        C_all = C_T.flatten()
        P_all = P_T.flatten()
        valid = np.isfinite(C_all) & np.isfinite(P_all)
        C_valid = C_all[valid]
        P_valid = P_all[valid]
        coupling_1 = C_valid[P_valid > 0.5]
        coupling_0 = C_valid[P_valid <= 0.5]
        cl = self._coupling_label
        cs = self._coupling_short
        sns.kdeplot(coupling_1, label=r"High posterior node pairs")
        sns.kdeplot(coupling_0, label=r"Low posterior node pairs")
        if self.compare_to_mean_coupling:
            plt.xlabel(f"{cl} " + r"$\bar{C}_{ij}^{risk}$")
        else:
            plt.xlabel(f"{cl} " + r"$C_{ij}^{risk}$")
        plt.ylabel("Density")
        if self.compare_to_mean_coupling:
            plt.title(r"Average $\bar{C}_{ij}^{risk}$ conditioned on posterior")
        else:
            plt.title(r"$C_{ij}^{risk}$ conditioned on posterior")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.outdir / f"kde_{cs}_conditioned_on_posterior_binary.png")
        plt.savefig(self.outdir / f"kde_{cs}_conditioned_on_posterior_binary.svg")
        plt.show()
        self._save_hist(
            spearman_rho,
            title=f"Spearman: {cl} vs posterior",
            xlabel=r"Spearman $\rho$",
            outpath=self.outdir / "hist_spearman_rho.png",
        )
        self._save_hist(
            pearson_r,
            title=f"Pearson: {cl} vs posterior",
            xlabel="Pearson r",
            outpath=self.outdir / "hist_pearson_r.png",
        )
        self._save_hist(
            kendall_tau_arr,
            title=f"Kendall tau: {cl} vs posterior",
            xlabel="Kendall tau",
            outpath=self.outdir / "hist_kendall_tau.png",
        )
        self._save_hist(
            topk_overlap_arr,
            title=f"Top-k overlap: {cl} vs posterior",
            xlabel=f"Top-{int(self.topk_frac * 100)}% overlap fraction",
            outpath=self.outdir / f"hist_topk_overlap_{int(self.topk_frac * 100)}pct.png",
            vline=self.topk_frac,
            vline_label=f"Random baseline ({self.topk_frac:.0%})",
        )
        self._save_hist(
            roc_auc_arr,
            title=f"ROC-AUC: posterior predicts strong {cl} (>= p{int(self.strong_label_percentile)})",
            xlabel="ROC-AUC",
            outpath=self.outdir / f"hist_roc_auc_p{int(self.strong_label_percentile)}.png",
        )
        self._save_hist(
            ap_arr,
            title=f"Avg Precision: posterior predicts strong {cl} (>= p{int(self.strong_label_percentile)})",
            xlabel="Average Precision",
            outpath=self.outdir / f"hist_avg_precision_p{int(self.strong_label_percentile)}.png",
        )
        self._save_hist(
            mi_arr,
            title=f"Mutual information: posterior vs {cl}",
            xlabel="MI",
            outpath=self.outdir / "hist_mutual_info.png",
        )
        self._save_hist(
            edge_corr,
            title=f"Temporal Pearson: Risk coupling vs posterior",
            xlabel="Temporal Pearson r",
            outpath=self.outdir / "hist_edgewise_temporal_pearson.png",
        )
        self._save_hist(
            edge_spearman,
            title=f"Edge-wise Spearman correlation over time",
            xlabel=r"Temporal Spearman $\rho$",
            outpath=self.outdir / "hist_edgewise_temporal_spearman.png",
        )
        #self._save_scatter(
        #    values=(P_T.flatten(), C_T.flatten()),
        #    title=f"Scatter: posterior existence vs {cl} (all timesteps and edges)",
        #    xlabel="Posterior existence probability",
        #    ylabel=cl.capitalize(),
        #    outpath=self.outdir / "scatter_posterior_vs_coupling.png",
        #)

        # Print where saved
        print(f"Saved metrics and plots to: {self.outdir.absolute()}")


def _setup(checkpoint_path: str, checkpoint_name: str, env_name_override: str) -> tuple[RllibAgent, Environment, CustomizedGrid2OpEnvironment]:
    """
    Returns an agent, env, and env-gym-wrapper given the
    :param checkpoint_path: path under which all checkpoints of a training sessions are stored
    :param checkpoint_name: the specific checkpoint for example checkpoint_000010
    :param env_name_override: the env to use (override to run agent on validation or test env)
    :return: rllib_agent, g2op_env, gym_wrapper
    """
    config = load_config(checkpoint_path)
    env_config = config["evaluation_config"]["env_config"]
    if env_name_override:
        env_config["env_name"] = env_name_override

    env_name = env_config["env_name"]

    logger.info(f"Loading RAPPO from: {checkpoint_path}")
    logger.info(f"Checkpoint: {checkpoint_name}")
    logger.info(f"Environment: {env_name}")

    # Use existing load_rllib_agent function
    rllib_agent, g2op_env, gym_wrapper = load_rllib_agent(
        checkpoint_path=checkpoint_path,
        policy_name="reinforcement_learning_policy",
        checkpoint_name=checkpoint_name,
        env_name=env_name,
        env_config=env_config
    )
    return rllib_agent, g2op_env, gym_wrapper


def main():
    plt.show()
    checkpoint_path = "/home/adrian/Schreibtisch/1901/1901_rappo_with_anneal_different_betas/CustomPPO_0_426b7_2026-01-19_10-28-48"
    checkpoint_name = "checkpoint_000020"
    env_name_override = "l2rpn_case14_sandbox_test"

    agent, env, gym_env = _setup(checkpoint_path, checkpoint_name, env_name_override)
    analysis_agent = LatentGraphAnalysisAgent(agent, gym_env, Hypothesis2verifier(compare_to_mean_coupling=False))
    logger.info("Agent loaded! Starting episodes...\n")
    analysis_agent.analyze(num_episodes=50)


if __name__ == "__main__":
    main()