"""
"""

__all__ = ["PlotGvfIkCons"]

import numpy as np
from collections.abc import Iterable

# Graphic tools
import matplotlib.pyplot as plt

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import fixedwing_patch, config_data_axis

A_FIT = 1.35

#######################################################################################

class PlotGvfIkCons:
    def __init__(self, data, gvf_traj, **kwargs):
        self.data = data
        self.gvf_traj = gvf_traj

        # Default visual properties
        kw_fig = {
            "dpi": 100,
            "figsize": (10,4)
        }

        kw_ax = {
            "x_step": 100,
            "y_step": 50,
            "y_right": False,
            "xlims": None,
            "ylims": None  
        }
        
        kw_patch = {
            "fc": "None",
            "ec": "red",
            "size": 15,
            "lw": 1,
            "zorder": 3,
        }

        # Update defaults with user-specified values
        self.kw_fig = parse_kwargs(kwargs, kw_fig)
        self.kw_ax = parse_kwargs(kwargs, kw_ax)
        self.kw_patch = parse_kwargs(kwargs, kw_patch)

        # Create subplots
        self.fig_gvf, self.ax_gvf = plt.subplots(**self.kw_fig)

    # ---------------------------------------------------------------------------------
    def config_axes(self):
        self.ax_gvf.set_xlabel(r"$X$ [m]")
        self.ax_gvf.set_ylabel(r"$Y$ [m]")
        self.ax_gvf.set_aspect("equal")
        config_data_axis(self.ax_gvf, **self.kw_ax)

    def plot(self, num_patches=2, **kw_line):
        self.config_axes()

        # Extract derired data
        x = np.array(self.data["p"].tolist())[1:,:,0]
        y = np.array(self.data["p"].tolist())[1:,:,1]
        theta = np.array(self.data["theta"].tolist())

        gvf_s = np.array(self.data["s"].tolist())[0]
        gvf_ke = np.array(self.data["ke"].tolist())[0]

        N = x.shape[1]
        idx_list = np.linspace(0, x.shape[0]-1, num_patches, dtype=int)

        # ------------------------------------------------
        # Plot the robots
        self.ax_gvf.plot(x, y, "b", **kw_line)

        for idx in idx_list:
            for i in range(N):
                patch = fixedwing_patch([x[idx,i], y[idx,i]], theta[idx,i], 
                                             **self.kw_patch)
                self.ax_gvf.add_artist(patch)

        # Plot the GVF
        if isinstance(self.gvf_traj, Iterable):
            for i in range(len(self.gvf_traj)):
                self.gvf_traj[i].gen_vector_field(area=1000, s=gvf_s, ke=gvf_ke)
                self.gvf_traj[i].draw(self.fig_gvf, self.ax_gvf, lw=1.4, draw_field=False)
        else:
                self.gvf_traj.gen_vector_field(area=1000, s=gvf_s, ke=gvf_ke)
                self.gvf_traj.draw(self.fig_gvf, self.ax_gvf, lw=1.4, draw_field=False)

        return self.ax_gvf
        
        # ------------------------------------------------

    def save(self, filename, dpi=100):
        plt.savefig(filename, dpi=dpi)

#######################################################################################