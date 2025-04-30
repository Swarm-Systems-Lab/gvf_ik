"""
"""

__all__ = ["Oscillator"]

import numpy as np

from ssl_simulator.controllers import Controller
from .utils import CircularBuffer

#######################################################################################

class Oscillator(Controller):
    def __init__(self, A, omega, speed, buff_len = 5000):

        # Controller settings
        self.A = A
        self.omega = omega
        self.speed = speed
        
        self.period = 2*np.pi / self.omega[0]

        # Other variables for logging
        self.pos_buff = CircularBuffer(int(buff_len), self.period)

        # ---------------------------
        # Controller output variables
        self.control_vars = {
            "u": None,
        }

        # Controller variables to be tracked by logger
        self.tracked_vars = {
            "speed": self.speed,
            "gamma_A": self.A,
            "gamma_omega": self.omega,
            "gamma": None,
            "x_dot": None,
            "x_dot_avg": None
        }

        # Controller data
        self.init_data()

    def compute_control(self, time, state):
        """
        Follow y = gamma(t) = A * sin(w t) at constant speed ||v|| = s
        """
        num_agents = state["p"].shape[0]

        # Validate that the behavior parameters satisfy speed constraints
        max_speed_violation = (self.A * self.omega > self.speed)
        if max_speed_violation.any():
            print(max_speed_violation)
            raise ValueError("A * omega should be <= speed!")

        # Compute the desired trajectory position and velocity
        gamma = self.A * np.sin(self.omega * time)
        gamma_dot = self.A * self.omega * np.cos(self.omega * time)
        x_dot = np.sqrt(self.speed**2 - gamma_dot**2)
        self.tracked_vars["x_dot"] = x_dot

        # Store trajectory information
        self.tracked_vars["gamma"] = gamma

        # Initialize and update control inputs
        self.control_vars["u"] = np.zeros((num_agents, 2))
        self.control_vars["u"][:, 0] = x_dot
        self.control_vars["u"][:, 1] = gamma_dot

        # Log parametric velocity
        self.pos_buff.enqueue(time, x_dot)
        valid_x_dot_values = np.array(self.pos_buff.get_valid_items())

        # Compute averaged parametric velocity 
        # (TODO: Apply weighted mean by dt in practice)
        x_dot_avg = np.mean(valid_x_dot_values, axis=0)
        self.tracked_vars["x_dot_avg"] = x_dot_avg

        return self.control_vars
    
    #######################################################################################