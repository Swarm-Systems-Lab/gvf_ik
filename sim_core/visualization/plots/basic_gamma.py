"""
"""

__all__ = ["PlotBasicGamma"]

import numpy as np
import matplotlib.pyplot as plt

#######################################################################################

class PlotBasicGamma:
    def __init__(self, data):
        self.data = data
        self.fig, self.ax = plt.subplots()

    # ---------------------------------------------------------------------------------
    def config_axes(self):
        self.ax.set_xlabel(r"$X$ [m]")
        self.ax.set_ylabel(r"$Y$ [m]")
        self.ax.grid(True)

    def plot(self):
        self.config_axes()
        
        # Extract derired data
        x = self.data["p"][:,:,0]
        y = self.data["p"][:,:,1]
        gamma = self.data["gamma"][:,:]

        # Create the plot
        self.ax.plot(x, y, "b", zorder=3)
        self.ax.plot(x, y[0,:] + gamma, "--k", alpha=1, lw=1)
        self.ax.scatter(x[0,:], y[0,:], edgecolors="r", marker="s", facecolors="None")
        self.ax.scatter(x[-1,:], y[-1,:],edgecolors="b", facecolors="None")

        return self.ax
    
    def save(self, filename, dpi=100):
        plt.savefig(filename, dpi=dpi)

#######################################################################################