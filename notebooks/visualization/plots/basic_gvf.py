"""
"""

__all__ = ["PlotBasicGvf"]

import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Iterable

from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import Plotter, vector2d, config_axis
from ssl_simulator.components.gvf import GvfTrajectoryPlotter

#######################################################################################

class PlotBasicGvf:
    def __init__(self, data, gvf_traj, **kwargs):
        self.data = data
        self.gvf_traj = gvf_traj
        self.kw_ax = kwargs

        # Create subplots
        self.fig, self.ax = plt.subplots()

    # ---------------------------------------------------------------------------------
    def config_axes(self):
        self.ax.set_xlabel(r"$X$ [m]")
        self.ax.set_ylabel(r"$Y$ [m]")
        self.ax.grid(True)
        self.ax.set_aspect("equal")
        config_axis(self.ax, **self.kw_ax)

    def plot(self, alpha=1):
        self.config_axes()
        
        # Extract derired data
        x = np.array(self.data["p"].tolist())[:,:,0]
        y = np.array(self.data["p"].tolist())[:,:,1]
        s = np.array(self.data["s"].tolist())[0]
        ke = np.array(self.data["ke"].tolist())[0]

        # Plot the robots
        self.ax.plot(x, y, "b", alpha=alpha)
        self.ax.scatter(x[0,:], y[0,:], edgecolors="r", marker="s", facecolors="None")
        self.ax.scatter(x[-1,:], y[-1,:],edgecolors="r", facecolors="None", zorder=3)

        # Plot the GVF
        if isinstance(self.gvf_traj, Iterable):
            for i in range(len(self.gvf_traj)):
                gvf_traj_plotter = GvfTrajectoryPlotter(self.gvf_traj[i], self.fig, self.ax)
                gvf_traj_plotter.draw(lw=1.4, draw_field=False)
        else:   
                gvf_traj_plotter = GvfTrajectoryPlotter(self.gvf_traj, self.fig, self.ax)
                gvf_traj_plotter.draw(lw=1.4, draw_field=False)

        return self.ax
    
    def save(self, filename, dpi=100):
        plt.savefig(filename, dpi=dpi)

#######################################################################################