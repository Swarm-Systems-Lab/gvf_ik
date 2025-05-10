"""
"""

__all__ = ["PlotBasicGvf"]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ssl_simulator import parse_kwargs, load_class
from ssl_simulator.visualization import config_axis, fixedwing_patch
from ssl_simulator.components.gvf import GvfIkTrajectoryPlotter

#######################################################################################

class PlotBasicGvf:
    def __init__(self, data, settings, **kwargs):
        self.data = data
        self.gvf_traj = load_class(
            "ssl_simulator.components.gvf",
            settings["gvf_traj"]["__class__"], **settings["gvf_traj"]["__params__"]
        )
        self.kw_ax = kwargs

        # Default visual properties
        kw_fig = dict(dpi=100, figsize=(6,9))
        kw_ax = dict(xlims=None, ylims=None)
        self.kw_patch = dict(fc="royalblue", ec="black", size=15, lw=1, zorder=3)

        # Update defaults with user-specified values
        self.kw_fig = parse_kwargs(kwargs, kw_fig)
        self.kw_ax = parse_kwargs(kwargs, kw_ax)

        # Create subplots
        self.fig, self.ax = plt.subplots(**self.kw_fig)

    # ---------------------------------------------------------------------------------
    def config_axes(self):
        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$ [L]")
        self.ax.grid(True)
        self.ax.set_aspect("equal")
        config_axis(self.ax, **self.kw_ax)

    def plot(self, fw=False, draw_field=False, **kwargs):
        self.config_axes()
        
        # Extract derired data
        x = self.data["p"][:,:,0]
        y = self.data["p"][:,:,1]
        s = self.data["s"][-1]
        ke = self.data["ke"][-1]
        speed = self.data["speed"][-1,0]
        gamma = self.data["gamma"][-1,0]
        gamma_dot = self.data["gamma_dot"][-1,0]

        # Plot the robots
        if fw:
            theta = self.data["theta"]
            for i in range(x.shape[1]):
                patch_i = fixedwing_patch([x[0,i], y[0,i]], theta[0,i], **self.kw_patch)
                patch_f = fixedwing_patch([x[-1,i], y[-1,i]], theta[-1,i], **self.kw_patch)
                patch_i.set_alpha(0.5)
                self.ax.add_artist(patch_i)  
                self.ax.add_artist(patch_f)     
        else:
            self.ax.scatter(x[0,:], y[0,:], edgecolors="r", marker="s", facecolors="None")
            self.ax.scatter(x[-1,:], y[-1,:],edgecolors="r", facecolors="None", zorder=3)
        self.ax.plot(x, y, "royalblue")

        # Plot the GVF
        kw_field = dict(color="blue", alpha=0.8, zorder=1, lw=1.4, pts=30,
                        s=s, ke=ke, gamma=gamma, gamma_dot=gamma_dot, speed=speed)
        kw_field = parse_kwargs(kwargs, kw_field)
 
        gvf_traj_plotter = GvfIkTrajectoryPlotter(self.gvf_traj, self.fig, self.ax)
        gvf_traj_plotter.draw(draw_field=draw_field, **kw_field)

        # Plot the legend
        if draw_field:
            try:
                # Ensure LaTeX rendering is active
                plt.rcParams['text.usetex'] = True
                plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

                # Legend elements
                arr = self.ax.scatter([], [], c=kw_field["color"], alpha=kw_field["alpha"],
                                      marker=r"$\uparrow$", s=60, label=r"$f(\phi(p),t_f)$")
                reg = mpatches.Patch(color='grey', alpha=0.2, label=r"$\{p \in \mathbb{R}^2 \; : \; \|\vartheta(p)\| < v\}$")

                self.ax.legend(handles=[arr, reg], fancybox=True, prop={"size": 10}, ncols=2, loc="upper left")

            except Exception as e:
                # Latex rendering failed
                print(f"LaTeX rendering failed: {e}\nFalling back to non-LaTeX labels. Make sure LaTeX and amssymb are installed.")

                # Fallback without LaTeX
                plt.rcParams['text.usetex'] = False
                arr = self.ax.scatter([], [], c=kw_field["color"], alpha=kw_field["alpha"],
                                      marker=r"$\uparrow$", s=60, label="f(\phi(p),t_f)")
                reg = mpatches.Patch(color='grey', alpha=0.2, label="{p in R² : ||\vartheta(p)|| < v}")

                self.ax.legend(handles=[arr, reg], fancybox=True, prop={"size": 10}, ncols=2, loc="upper left")

        return self.ax
    
    def save(self, filename, dpi=100):
        plt.savefig(filename, dpi=dpi)

#######################################################################################