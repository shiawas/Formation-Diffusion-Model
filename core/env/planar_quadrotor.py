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
        """
        dynamics with JAX-compatible code.

        Equations are from the Aerial Robotics coursera lecture
        https://www.coursera.org/lecture/robotics-flight/2-d-quadrotor-control-kakc6
        """
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
        x_dot_new = u1 * jnp.cos(phi)
        y_dot_new = u1 * jnp.sin(phi)
        phi_dot_new = u2

        x_new = x + x_dot_new * dt
        y_new = y + y_dot_new * dt
        phi_new = phi + phi_dot_new * dt

        next_state = jnp.array([x_new, x_dot_new, y_new, y_dot_new, phi_new, phi_dot_new], dtype=float)

        self.state = next_state
        return next_state