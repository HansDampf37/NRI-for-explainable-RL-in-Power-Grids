from pprint import pprint

from ray.rllib.algorithms import DQNConfig

from common import Grid2OpEnvWrapper


def main():
    config = (
        DQNConfig()
        .environment(Grid2OpEnvWrapper)
        .framework("torch")
        .training(
            dueling=True,
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 60000,
                "alpha": 0.5,
                "beta": 0.5,
            }
        )
        .env_runners(num_env_runners=1)
        .evaluation(evaluation_interval=10)
    )

    algo = config.build_algo()
    for _ in range(100):
        pprint(algo.train())


if __name__ == '__main__':
    main()