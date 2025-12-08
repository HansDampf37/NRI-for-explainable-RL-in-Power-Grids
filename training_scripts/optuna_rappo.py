import os

import hydra
import optuna
from optuna.trial import BaseTrial

from src.ra_agents.ppo import train_relations_aware_ppo


def objective(trial: BaseTrial):
    with hydra.initialize(config_path="../hydra_configs", version_base="1.3"):
        cfg = hydra.compose(config_name="config")
        # Allow quick overrides via env var for smoke tests
        cfg.rl.train.timesteps = int(os.getenv("OPTUNA_TIMESTEPS", "50000"))
        cfg.rl.model.prior_for_graph_edges_existing = trial.suggest_float('prior', 0.0, 1.0)
        cfg.rl.ppo.sb3.kl_coef = trial.suggest_float('kl_coef', 0.0, 1.0)
        cfg.rl.model.temperature = trial.suggest_float('temperature', 0.0, 1.0)
        hidden_out_dim = trial.suggest_int('hidden_dim', 32, 128)
        cfg.rl.model.features_extractor_kwargs.hidden_dim = hidden_out_dim
        cfg.rl.model.features_extractor_kwargs.out_dim = hidden_out_dim
        cfg.rl.model.features_extractor_kwargs.num_layers = trial.suggest_int('num_layers', 2, 4)
        results = train_relations_aware_ppo(cfg)
        val_sd = results["rl_algorithm"]["val"]["survival_duration"]
        return float(sum(val_sd) / len(val_sd)) if len(val_sd) > 0 else 0.0


study = optuna.create_study(study_name="optimize_encoder_hps",direction="maximize")
study.optimize(objective, n_trials=20)

print(study.best_params)