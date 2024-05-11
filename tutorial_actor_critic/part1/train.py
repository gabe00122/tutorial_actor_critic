import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import jax
import numpy as np
import optax
from flax import linen as nn
from jax import random
from jax.typing import ArrayLike
import os

from .actor_critic import HyperParameters, ModelUpdateParams, ActorCritic, TrainingState
from ..mlp import MlpBody, ActorHead, CriticHead


def main(seed: int = 0) -> list[float]:
    env_name = 'CartPole-v1'
    total_steps = 1000000
    hyper_parameters = HyperParameters(
        actor_learning_rate=optax.linear_schedule(0.0002, 0.0, total_steps),
        critic_learning_rate=optax.linear_schedule(0.001, 0.0, total_steps),
        discount=optax.constant_schedule(0.99),
    )

    env = gym.make(env_name)  # render_mode='human'
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]

    rng_key = random.PRNGKey(seed)

    # Create models
    actor_critic = create_actor_critic(hyper_parameters, action_space)
    jit_actor_critic(actor_critic)

    # Initialize params
    rng_key, actor_critic_key = random.split(rng_key)
    training_state = actor_critic.init(state_space, actor_critic_key)

    # metrics
    total_reward = 0
    total_rewards = []

    obs, _ = env.reset()
    for step in range(total_steps):
        rng_key, action_key = random.split(rng_key)
        action = actor_critic.sample_action(training_state, obs, action_key)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        model_update_params = ModelUpdateParams(
            step=step,
            obs=obs,
            actions=action,
            rewards=reward,
            next_obs=next_obs,
            done=done,
        )

        training_state, metrics = actor_critic.update_models(training_state, model_update_params)
        obs = next_obs

        total_reward += reward

        if done:
            obs, _ = env.reset()
            total_rewards.append(total_reward)
            total_reward = 0

            if len(total_rewards) % 100 == 99:
                print(total_rewards[-1])

    record_video("videos/rl-video", env_name, actor_critic, training_state, rng_key, 10)

    return total_rewards


def record_video(file_path: str, env_name: str, actor_critic: ActorCritic, training_state: TrainingState, rng_key: ArrayLike, episodes: int):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        os.path.dirname(file_path),
        episode_trigger=lambda _: True,
        name_prefix=os.path.basename(file_path))

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            rng_key, action_key = random.split(rng_key)
            action = actor_critic.sample_action(training_state, obs, action_key)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated


def create_actor_critic(hyper_parameters: HyperParameters, action_space: int) -> ActorCritic:
    actor_model = nn.Sequential([
        MlpBody(features=[64, 64]),
        ActorHead(actions=action_space),
    ])
    critic_model = nn.Sequential([
        MlpBody(features=[64, 64]),
        CriticHead(),
    ])

    return ActorCritic(actor_model, critic_model, hyper_parameters)


def jit_actor_critic(actor_critic: ActorCritic):
    """
    Jits an actor critic in place
    """
    actor_critic.init = jax.jit(actor_critic.init, static_argnames="state_space")
    actor_critic.sample_action = jax.jit(actor_critic.sample_action)
    actor_critic.update_models = jax.jit(actor_critic.update_models)


if __name__ == '__main__':
    for index in range(1):
        print(f"Starting training run {index}")
        total_rewards = main(index)
        # np_data = np.array(total_rewards, dtype=np.float32)
        # np.save(f"metrics/results_{index}", np_data)
