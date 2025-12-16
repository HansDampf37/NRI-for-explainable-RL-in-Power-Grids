import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import optuna
from omegaconf import DictConfig
from optuna.trial import BaseTrial

from src.common.constants import set_experiment_name
from src.ra_agents.ppo import train_relations_aware_ppo

_cfg: Optional[DictConfig] = None
logger = logging.getLogger(__name__)


def objective(trial: BaseTrial):
    # Allow quick overrides via env var for smoke tests
    _cfg.rl.train.timesteps = int(os.getenv("OPTUNA_TIMESTEPS", _cfg.optuna.train_duration_per_trial))
    _cfg.rl.model.prior_for_graph_edges_existing = trial.suggest_float('prior', 0.0, 1.0)
    _cfg.rl.ppo.sb3.kl_coef = trial.suggest_float('kl_coef', 0.0, 1.0)
    _cfg.rl.model.temperature = trial.suggest_float('temperature', 0.0, 1.0)
    hidden_out_dim = trial.suggest_int('hidden_dim', 32, 128)
    _cfg.rl.model.features_extractor_kwargs.hidden_dim = hidden_out_dim
    _cfg.rl.model.features_extractor_kwargs.out_dim = hidden_out_dim
    _cfg.rl.model.features_extractor_kwargs.num_layers = trial.suggest_int('num_layers', 2, 4)
    _cfg.rl.verbose=False
    _cfg.rl.train.curriculum_level_config.active = False # disable curriculum for optuna
    _cfg.rl.eval.during_training.active = False # disable intermediate evaluations for optuna
    logger.info(f"Starting trial {trial.number} with:"
                f"\n\tPrior: {_cfg.rl.model.prior_for_graph_edges_existing}"
                f"\n\tKL_ceof: {_cfg.rl.ppo.sb3.kl_coef}"
                f"\n\tTemperature: {_cfg.rl.model.temperature}"
                f"\n\thidden_out_dim: {hidden_out_dim}"
                f"\n\tnum_layers: {_cfg.rl.model.features_extractor_kwargs.num_layers}")
    results = train_relations_aware_ppo(_cfg)
    val_sd = results["rl_algorithm"]["val"]["survival_duration"]
    return float(sum(val_sd) / len(val_sd)) if len(val_sd) > 0 else 0.0


@hydra.main(config_path="../configs", config_name="configs", version_base="1.3")
def main(cfg: DictConfig):
    global _cfg
    _cfg = cfg
    logger.info(cfg)
    set_experiment_name(cfg.experiment_name)
    from src.common.constants import LOGS_PATH
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')[:-3]
    db_path = Path(LOGS_PATH, f"optuna_{timestamp}.db")
    logger.info(f"Optuna session started. Run `optuna-dashboard sqlite:///{db_path}` to examine the results")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    storage_url = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name="optimize_encoder_hps",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
    )

    timeout = _parse_timeout_to_seconds(cfg.timeout)
    if timeout is not None:
        logger.info(f"Set optimizing timeout of {cfg.timeout} ({timeout} seconds)")

    study.optimize(objective, n_trials=cfg.optuna.n_trials, timeout=timeout, n_jobs=cfg.optuna.n_jobs)
    print(study.best_params)


def _parse_timeout_to_seconds(timeout_str: str | None) -> float | None:
    if not timeout_str:
        return None

    h, m, s = map(int, timeout_str.split(":"))
    return float(h * 3600 + m * 60 + s)


if __name__ == "__main__":
    main()
