from jax import Array, numpy as jnp
from flax import linen as nn
from collections.abc import Callable, Sequence


class MlpBody(nn.Module):
    features: Sequence[int]
    activation: Callable[[Array], Array] = nn.relu

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        for i, feat in enumerate(self.features):
            x = nn.Dense(
                feat,
                name=f"mlp_layer_{i}",
                kernel_init=nn.initializers.he_normal(),
            )(x)
            x = self.activation(x)
        return x


class ActorHead(nn.Module):
    actions: int

    @nn.compact
    def __call__(self, inputs):
        actor_logits = nn.Dense(
            self.actions,
            name="actor_head",
            kernel_init=nn.initializers.variance_scaling(
                2.0 / 100, "fan_in", "truncated_normal"
            ),
        )(inputs)

        # if mask is not None:
        #     actor_logits = jnp.where(mask, actor_logits, -jnp.inf)

        return actor_logits


class CriticHead(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        value = nn.Dense(
            1,
            name="critic_head",
            kernel_init=nn.initializers.variance_scaling(
                2.0 / 100, "fan_in", "truncated_normal"
            ),
        )(inputs)
        return jnp.squeeze(value)
