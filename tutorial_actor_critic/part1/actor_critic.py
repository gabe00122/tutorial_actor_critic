import jax
from jax import random, numpy as jnp
from jax.typing import ArrayLike
from flax import linen as nn
from optax import Schedule
from typing import NamedTuple, Any


class TrainingState(NamedTuple):
    importance: ArrayLike
    actor_params: Any
    critic_params: Any


class Metrics(NamedTuple):
    td_error: float
    state_value: float


class HyperParameters(NamedTuple):
    discount: Schedule
    actor_learning_rate: Schedule
    critic_learning_rate: Schedule


class ModelUpdateParams(NamedTuple):
    step: ArrayLike

    # Just remember SARS(A)
    obs: ArrayLike  # S
    actions: ArrayLike  # A
    rewards: ArrayLike  # R
    next_obs: ArrayLike  # S
    done: ArrayLike


class ActorCritic:
    def __init__(
        self,
        actor_model: nn.Module,
        critic_model: nn.Module,
        hyper_parameters: HyperParameters,
    ):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.hyper_parameters = hyper_parameters

    def init(self, state_space: int, key: ArrayLike) -> TrainingState:
        dummy_state = jnp.zeros((state_space,), dtype=jnp.float32)

        key, actor_key, critic_key = random.split(key, 3)
        actor_params = self.actor_model.init(actor_key, dummy_state)
        critic_params = self.critic_model.init(critic_key, dummy_state)

        training_state = TrainingState(
            importance=jnp.float32(1.0),
            actor_params=actor_params,
            critic_params=critic_params,
        )
        return training_state

    def sample_action(
        self, training_state: TrainingState, obs: ArrayLike, key: ArrayLike
    ):
        logits = self.actor_model.apply(training_state.actor_params, obs)
        return random.categorical(key, logits)

    def action_log_probability(self, actor_params, obs: ArrayLike, action: ArrayLike):
        logits = self.actor_model.apply(actor_params, obs)
        return nn.log_softmax(logits)[action]

    def update_models(
        self, training_state: TrainingState, params: ModelUpdateParams
    ) -> tuple[TrainingState, Metrics]:
        actor_params = training_state.actor_params
        critic_params = training_state.critic_params
        importance = training_state.importance
        step = params.step

        # Just remember SARSA expect we skip the final action here
        obs = params.obs  # State
        actions = params.actions  # Action
        rewards = params.rewards  # Reward
        next_obs = params.next_obs  # State
        done = params.done  # Also State

        # Let's calculate are hyperparameters from the schedule
        discount = self.hyper_parameters.discount(step)
        actor_learning_rate = self.hyper_parameters.actor_learning_rate(step)
        critic_learning_rate = self.hyper_parameters.critic_learning_rate(step)

        # Calculate the TD error
        state_value = self.critic_model.apply(critic_params, obs)
        td_error = rewards - state_value
        td_error += jax.lax.cond(
            done,
            lambda: 0.0,  # if the episode is over our next predicted reward is always zero
            lambda: discount * self.critic_model.apply(critic_params, next_obs)
        )

        # Update the critic
        critic_gradient = jax.grad(self.critic_model.apply)(critic_params, obs)
        critic_params = update_params(
            critic_params,
            critic_gradient,
            critic_learning_rate * td_error,
        )

        # Update the actor
        actor_gradient = jax.grad(self.action_log_probability)(actor_params, obs, actions)
        actor_params = update_params(
            actor_params,
            actor_gradient,
            actor_learning_rate * importance * td_error,
        )

        # Record Metrics
        metrics = Metrics(
            td_error=td_error,
            state_value=state_value,
        )

        importance = jax.lax.cond(
            done,
            lambda: 1.0,
            lambda: importance * discount
        )

        return TrainingState(
            importance=importance,
            actor_params=actor_params,
            critic_params=critic_params
        ), metrics


def update_params(params, grad, step_size):
    return jax.tree_map(
        lambda param, grad_param: param + step_size * grad_param,
        params, grad
    )
