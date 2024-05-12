import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from jax import random
from jax.typing import ArrayLike
from .actor_critic import ActorCritic, TrainingState


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
