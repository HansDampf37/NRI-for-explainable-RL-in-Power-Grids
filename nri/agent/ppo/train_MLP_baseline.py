import os.path
import uuid
from datetime import datetime
from pathlib import Path

import grid2op
import hydra
from grid2op.gym_compat import BoxGymObsSpace
from hydra.utils import instantiate
from lightsim2grid import LightSimBackend
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO

from baselines.baseline_agent import BaselineAgent, evaluate_agent
from common import G2OpGymEnv
from common.constants import LOGS_PATH, MODELS_PATH, EVAL_PATH
from common.rewards import MazeRLReward
from nri.agent.ppo.PPOTopoPolicy import Sb3PPOTopologyPolicy


def get_env(cfg) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action space from hydra config. Observation space is always
    BoxGymObsSpace since we will use an MLP instead of a GNN

    :param cfg: The hydra config
    :return: The environment
    """
    env: G2OpGymEnv = instantiate(
        cfg.env.training_env,
        obs_space_creation=lambda e: BoxGymObsSpace(grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.rl.act_space, grid2op_action_space=e.action_space)
    )
    return env


@hydra.main(config_path="../../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    group = "relation-unaware-baselines/ppo"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    name = f"mlp_{timestamp}_{uuid.uuid4().hex}"
    env = get_env(cfg)

    algorithm = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=cfg.rl.ppo.sb3.verbose,
        learning_rate=cfg.rl.ppo.sb3.learning_rate,
        n_steps=cfg.rl.ppo.sb3.n_steps,
        batch_size=cfg.rl.ppo.sb3.batch_size,
        n_epochs=cfg.rl.ppo.sb3.n_epochs,
        gamma=cfg.rl.ppo.sb3.gamma,
        gae_lambda=cfg.rl.ppo.sb3.gae_lambda,
        clip_range=cfg.rl.ppo.sb3.clip_range,
        clip_range_vf=cfg.rl.ppo.sb3.clip_range_vf,
        normalize_advantage=cfg.rl.ppo.sb3.normalize_advantage,
        ent_coef=cfg.rl.ppo.sb3.ent_coef,
        vf_coef=cfg.rl.ppo.sb3.vf_coef,
        max_grad_norm=cfg.rl.ppo.sb3.max_grad_norm,
        use_sde=cfg.rl.ppo.sb3.use_sde,
        sde_sample_freq=cfg.rl.ppo.sb3.sde_sample_freq,
        tensorboard_log=os.path.join(LOGS_PATH, group),
        policy_kwargs={
            "net_arch": cfg.rl.ppo.sb3.policy_kwargs.net_arch,
        }
    )
    algorithm.learn(total_timesteps=600000, tb_log_name=name, log_interval=cfg.rl.train.log_interval)
    algorithm.save(os.path.join(MODELS_PATH, group, name))

    # evaluate
    agent = BaselineAgent(
        env.g2op_action_space,
        Sb3PPOTopologyPolicy(algorithm),
        safe_max_rho=cfg.env.safe_max_rho,
    )
    for dataset in ["train", "test", "val"]:
        grid2op_env = grid2op.make(f"{cfg.env.env_name}_{dataset}", backend=LightSimBackend(), reward_class=MazeRLReward)
        evaluate_agent(
            agent=agent,
            env=grid2op_env,
            num_episodes=cfg.baseline.eval.nb_episodes,
            path_results=Path(os.path.join(EVAL_PATH, group, name + "_" + dataset))
        )


if __name__ == "__main__":
    main()
