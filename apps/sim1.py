import os

import numpy as np
import matplotlib.pyplot as plt

# Import the Swarm Systems Lab Simulator
from ssl_simulator import SimulationEngine, add_src_to_path, create_dir
from ssl_simulator.robot_models import Unicycle2D

from ssl_simulator.components.gvf import GvfEllipse

add_src_to_path(__file__)
from apps import AppGameGVFIK
from sim_core.visualization import PlotterGvfIK
from sim_core.controllers import GvfIK

def setup_sim1():
    # Define the initial state
    N = 1

    p = np.ones((N,2)) * np.array([[-200,-10]])
    speed = np.ones((N)) * np.array([14])
    theta = np.ones((N)) * np.array([-90])

    x0 = [p, speed, theta]

    # Controller settings
    A = np.ones((N)) * 0.3
    omega = np.ones((N)) * np.pi/4

    # --------------------------------
    # Generate the trajectory to be followed
    a, b = 60, 60
    XYoff, alpha = [0, 0], 0

    gvf_traj = GvfEllipse(XYoff,alpha,a,b)

    # Controller and simulator settings
    s, ke, kn = 1, 0.35, 1
    dt = 0.01

    # Construct the Simulator Engine
    robot_model = Unicycle2D(x0, omega_lims=[-np.pi/4, np.pi/4])
    controller = GvfIK(gvf_traj, s, ke, kn, A, omega)
    simulator_engine = SimulationEngine(robot_model, controller, time_step=dt, log_size=10)

    fig, ax = plt.subplots(dpi=200, figsize=(9,6))
    simulator_plotter = PlotterGvfIK(ax=ax, data=simulator_engine.data, settings=simulator_engine.settings, 
                                     xlims=[-220,140], ylims=[-120,120], tail_len=10)

    return fig, ax, simulator_engine, simulator_plotter

# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    app = AppGameGVFIK(*setup_sim1())
    # plt.show()