import gymnasium as gym
import numpy as np
import optax
from flax import linen as nn
from jax import random

from . import actor_critic
from .util import record_video
from ..mlp import MlpBody, ActorHead, CriticHead


def main(seed: int = 0) -> list[float]:
    env_name = 'CartPole-v1'
    total_steps = 800_000

    actor_learning_rate = optax.linear_schedule(0.0001, 0.0, total_steps)
    critic_learning_rate = optax.linear_schedule(0.0005, 0.0, total_steps)
    discount = 0.99
    actor_features = (64, 64)
    critic_features = (64, 64)

    env = gym.make(env_name)  # render_mode='human'
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]

    actor_model = nn.Sequential((
        MlpBody(features=actor_features),
        ActorHead(actions=action_space),
    ))
    critic_model = nn.Sequential((
        MlpBody(features=critic_features),
        CriticHead(),
    ))

    rng_key = random.PRNGKey(seed)

    # Initialize params
    rng_key, actor_critic_key = random.split(rng_key)
    training_state = actor_critic.create_training_state(actor_model, critic_model, state_space, actor_critic_key)

    # metrics
    total_reward = 0
    total_rewards = []

    obs, _ = env.reset()
    for step in range(total_steps):
        rng_key, action_key = random.split(rng_key)
        action = actor_critic.sample_action(actor_model, training_state, obs, action_key)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        update_args = actor_critic.UpdateArgs(
            actor_learning_rate=actor_learning_rate(step),
            critic_learning_rate=critic_learning_rate(step),
            discount=discount,
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
        )

        training_state = actor_critic.update_models(actor_model, critic_model, training_state, update_args)
        obs = next_obs

        total_reward += reward

        if done:
            obs, _ = env.reset()
            total_rewards.append(total_reward)
            total_reward = 0

            if len(total_rewards) % 100 == 99:
                print(total_rewards[-1])

    record_video("output/videos/rl-video", env_name, actor_model, training_state, rng_key, 10)

    return total_rewards


if __name__ == '__main__':
    for index in range(1):
        print(f"Starting training run {index}")
        total_rewards = main(index)
        np_data = np.array(total_rewards, dtype=np.float32)

        # Path("output/metrics_tanh").mkdir(parents=True, exist_ok=True)
        # np.save(f"output/metrics_tanh/results_{index}", np_data)
