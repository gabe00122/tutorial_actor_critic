from typing import NamedTuple, Any

import jax
from flax import linen as nn
from jax import random, numpy as jnp
from jax.typing import ArrayLike
from functools import partial


class TrainingState(NamedTuple):
    importance: ArrayLike
    actor_params: Any
    critic_params: Any


class UpdateArgs(NamedTuple):
    discount: float
    actor_learning_rate: float
    critic_learning_rate: float

    obs: ArrayLike
    action: ArrayLike
    reward: ArrayLike
    next_obs: ArrayLike
    done: ArrayLike


def create_training_state(
    actor_model, critic_model, state_space: int, key: ArrayLike
) -> TrainingState:
    dummy_state = jnp.zeros((state_space,), dtype=jnp.float32)

    key, actor_key, critic_key = random.split(key, 3)
    actor_params = actor_model.init(actor_key, dummy_state)
    critic_params = critic_model.init(critic_key, dummy_state)

    training_state = TrainingState(
        importance=jnp.float32(1.0),
        actor_params=actor_params,
        critic_params=critic_params,
    )
    return training_state


@partial(jax.jit, static_argnums=0)
def sample_action(actor_model, training_state, obs, rng_key):
    logits = actor_model.apply(training_state.actor_params, obs)
    return random.categorical(rng_key, logits)


def temporal_difference_error(critic_model, critic_params, update_args):
    state_value = critic_model.apply(critic_params, update_args.obs)
    next_state_value = jax.lax.cond(
        update_args.done,
        lambda: 0.0,
        lambda: update_args.discount
        * critic_model.apply(critic_params, update_args.next_obs),
    )

    estimated_reward = state_value - next_state_value
    td_error = update_args.reward - estimated_reward

    return td_error


def update_critic(critic_model, critic_params, update_args, td_error):
    critic_gradient = jax.grad(critic_model.apply)(critic_params, update_args.obs)
    critic_params = update_params(
        critic_params,
        critic_gradient,
        update_args.critic_learning_rate * td_error,
    )

    return critic_params


def action_log_probability(actor_model, actor_params, obs, action):
    logits = actor_model.apply(actor_params, obs)
    return nn.log_softmax(logits)[action]


def update_actor(actor_model, actor_params, update_args, td_error, importance):
    actor_gradient = jax.grad(action_log_probability, argnums=1)(
        actor_model, actor_params, update_args.obs, update_args.action
    )
    actor_params = update_params(
        actor_params,
        actor_gradient,
        update_args.actor_learning_rate * td_error * importance,
    )

    return actor_params


@partial(jax.jit, static_argnums=(0, 1))
def update_models(
    actor_model,
    critic_model,
    training_state: TrainingState,
    update_args: UpdateArgs,
) -> TrainingState:
    actor_params = training_state.actor_params
    critic_params = training_state.critic_params
    importance = training_state.importance

    td_error = temporal_difference_error(critic_model, critic_params, update_args)
    critic_params = update_critic(critic_model, critic_params, update_args, td_error)
    actor_params = update_actor(
        actor_model, actor_params, update_args, td_error, importance
    )

    importance = jax.lax.cond(
        update_args.done, lambda: 1.0, lambda: importance * update_args.discount
    )

    return TrainingState(
        importance=importance, actor_params=actor_params, critic_params=critic_params
    )


def update_params(params, grad, step_size):
    return jax.tree_map(
        lambda param, grad_param: param + step_size * grad_param, params, grad
    )
