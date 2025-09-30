from pathlib import Path

import grid2op
import hydra
from lightsim2grid import LightSimBackend
from omegaconf import DictConfig
from grid2op.Agent import RecoPowerlineAgent, DoNothingAgent

from baselines.baseline_agent import evaluate_agent


def evaluate(cfg: DictConfig):
    """
    Evaluate heuristic agents on our datasets.

    :param cfg: the hydra config
    """
    for dataset in ["train", "test", "val"]:
        env = grid2op.make(f"{cfg.env.env_name}_{dataset}", backend=LightSimBackend())
        for agent, name in zip([RecoPowerlineAgent(env.action_space), DoNothingAgent(env.action_space)],
                         ["reco_powerline_agent", "do_nothing_agent"]):
            evaluate_agent(
                agent=agent,
                env=env,
                num_episodes=cfg.eval.nb_episodes,
                path_results=Path("../data/evaluations/heuristic_agents").joinpath(name, dataset)
            )


@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()