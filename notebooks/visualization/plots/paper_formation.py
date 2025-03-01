"""
"""

__all__ = ["PlotFormation"]

import numpy as np
import matplotlib.pyplot as plt

# Graphic tools
import matplotlib.pyplot as plt

# Import tools from the Swarm Systems Lab Python Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import fixedwing_patch, config_data_axis

#######################################################################################

class PlotFormation:
    def __init__(self, **kwargs):

        # Default visual properties
        kw_fig = {
            "dpi": 100,
            "figsize": (9,3)
        }

        kw_ax = {
            "x_step": 10,
            "y_step": 10,
            "y_right": False,
            "xlims": [0,90],
            "ylims": [0,40]  
        }
        
        kw_patch = {
            "fc": "red",
            "ec": "k",
            "size": 6,
            "lw": 0.7,
            "zorder": 3,
        }

        # Update defaults with user-specified values
        self.kw_fig = parse_kwargs(kwargs, kw_fig)
        self.kw_ax = parse_kwargs(kwargs, kw_ax)
        self.kw_patch = parse_kwargs(kwargs, kw_patch)

        # Create subplots
        self.fig, self.axs = plt.subplots(1,2,**self.kw_fig)

        # ----------------
        # Parameters
        self.A1 = np.zeros(3) + 10
        self.A2 = np.array([0,10,0]) + 10
        self.B1 = np.zeros(3) + 80
        self.B2 = np.array([0,10,0]) + 70

        self.x1 = self.A1 + 35
        self.x2 = self.A2 + 40
        self.y = np.linspace(10,30,3)

        self.t = np.linspace(0,2*np.pi,100)

    def config_axes(self):
        # self.axs[0].set_ylabel(r"$Y$ [m]")
        for ax in self.axs:
            # ax.set_xlabel(r"$X$ [m]")
            ax.set_aspect("equal")
            config_data_axis(ax, **self.kw_ax)

            # Hide everything except the plot
            # ax.set_frame_on(False)  # Remove frame
            ax.set_xticks([])       # Remove x ticks
            ax.set_yticks([])       # Remove y ticks
            ax.set_xticklabels([])  # Remove x labels
            ax.set_yticklabels([])  # Remove y labels
            ax.xaxis.set_visible(False)  # Hide x-axis
            ax.yaxis.set_visible(False)  # Hide y-axis

    def plot(self):
        self.config_axes()

        ax1 = self.axs[0]
        ax2 = self.axs[1]

        for i in range(3):
            # ax1.add_artist(fixedwing_patch([self.x1[i], self.y[i]], 0, **self.kw_patch))
            # ax1.plot([self.A1[i], self.B1[i]], [self.y[i], self.y[i]], "k-")
            # ax1.plot(self.A1[i], self.y[i], "ks")
            # ax1.plot(self.B1[i], self.y[i], "ko")
            
            ax1.add_artist(fixedwing_patch([self.x2[i], self.y[i]], 0, **self.kw_patch))
            ax1.plot([self.A2[i], self.B2[i]], [self.y[i], self.y[i]], "k-")
            ax1.plot(self.A2[i], self.y[i], "ks")
            ax1.plot(self.B2[i], self.y[i], "ko")

            ax2.add_artist(fixedwing_patch([self.x1[i], self.y[i]], 0, **self.kw_patch))
            ax2.plot(45 + 0.5*30*np.cos(self.t), 0.5*20*np.sin(self.t), "k-")
            ax2.plot(45 + 30*np.cos(self.t), 20*np.sin(self.t), "k-")
            ax2.plot(45 + 1.5*30*np.cos(self.t), 1.5*20*np.sin(self.t), "k-")
        
        for i in range(2):
            kw_dashed = {"c":"k", "ls":"--", "lw":1, "zorder":2}
            ax1.plot([self.x2[i], self.x2[i+1]], [self.y[i], self.y[i+1]], **kw_dashed)
            ax2.plot([self.x1[i], self.x1[i+1]], [self.y[i], self.y[i+1]], **kw_dashed)
            

        return self.axs
    
    def save(self, filename, dpi=100):
        self.fig.savefig(filename, dpi=dpi)

#######################################################################################