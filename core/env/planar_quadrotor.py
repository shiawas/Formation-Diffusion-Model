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
        x_ddot = u1 * jnp.cos(phi)
        y_ddot = u1 * jnp.sin(phi)
        phi_ddot = u2

        next_state = (
            state
            + jnp.array(
                [
                    x_dot + x_ddot * dt,
                    x_ddot,
                    y_dot + y_ddot * dt,
                    y_ddot,
                    phi_dot + phi_ddot * dt,
                    phi_ddot,
                ]
            )
            * dt
        )

        self.state = next_state
        return next_state