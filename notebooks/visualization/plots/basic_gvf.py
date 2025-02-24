"""
"""

__all__ = ["PlotBasicGvf"]

import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Iterable

#######################################################################################

class PlotBasicGvf:
    def __init__(self, data, gvf_traj):
        self.data = data
        self.gvf_traj = gvf_traj

        # Create subplots
        self.fig, self.ax = plt.subplots()

    # ---------------------------------------------------------------------------------
    def config_axes(self):
        self.ax.set_xlabel(r"$X$ [m]")
        self.ax.set_ylabel(r"$Y$ [m]")
        self.ax.grid(True)
        self.ax.set_aspect("equal")

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
                self.gvf_traj[i].gen_vector_field(area=1000, s=s, ke=ke)
                self.gvf_traj[i].draw(self.fig, self.ax, lw=1.4, draw_field=False)
        else:
                self.gvf_traj.gen_vector_field(area=1000, s=s, ke=ke)
                self.gvf_traj.draw(self.fig, self.ax, lw=1.4, draw_field=False)

        return self.ax
    
    def save(self, filename, dpi=100):
        plt.savefig(filename, dpi=dpi)

#######################################################################################