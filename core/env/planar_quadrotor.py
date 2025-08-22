from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp


l_q = 1  # m, length of the quadrotor
w_q = 1


class PlanarQuadrotorEnv:
    def __init__(self, config: dict = None, state: Optional[jnp.ndarray] = None):
        if config is None:
            self.l_q = l_q
            self.w_q = w_q
        else:
            self.l_q = config["simulator"]["l_q"]
            self.w_q = config["simulator"]["w_q"]

        self.state: Optional[jnp.ndarray] = state

    @partial(jax.jit, static_argnums=0)
    def step(self, state=None, control=None, dt: float = 0.01):
        if state is None:
            state = self.state
            if state is None:
                raise Exception("state variable is not defined.")
            
        if control is None:
            control = jnp.array([0.0, 0.0])
        else:
            control = jnp.asarray(control)


        x, x_dot, y, y_dot, phi, phi_dot = state
        u1, u2 = control
        # Quadrotor dynamics
        x_dot = u1 * jnp.cos(phi)
        y_dot = u1 * jnp.sin(phi)
        phi_dot = u2

        x_next   = x + u1 * jnp.cos(phi) * dt
        y_next   = y + u1 * jnp.sin(phi) * dt
        phi_next = phi + u2 * dt
        next_state = jnp.array([
            x_next, x_dot,
            y_next, y_dot,
            phi_next, phi_dot
        ])
        self.state = next_state
        return next_state