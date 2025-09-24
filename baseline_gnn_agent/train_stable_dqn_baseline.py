from stable_baselines3 import DQN

from common import Grid2OpEnvWrapper


def main():
    env = Grid2OpEnvWrapper()
    dqn = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        train_freq=16,
        gradient_steps=8,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,
        target_update_interval=600,
        learning_starts=1000,
        buffer_size=10000,
        batch_size=128,
        learning_rate=4e-3,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="../data/logs/stable-baselines/dqn-mlp",
        seed=2,
    )
    dqn.learn(100, log_interval=10)
    dqn.save("../data/models/stable-baselines/dqn-mlp")

if __name__ == '__main__':
    main()