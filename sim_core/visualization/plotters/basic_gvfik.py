"""
"""

__all__ = ["PlotterGvfIK"]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import visualization tools from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs, load_class
from ssl_simulator.visualization import Plotter, config_axis
from ssl_simulator.components.gvf import PlotterGvfIk
from ssl_simulator.visualization import PlotterFixedwing

#######################################################################################

class PlotterGvfIK(Plotter):
    def __init__(self, ax, data, settings, **kwargs):
        self.ax = ax
        self.data = data
        
        if isinstance(settings["gvf_traj"], object):
            self.gvf_traj = settings["gvf_traj"]
        else:
            self.gvf_traj = load_class(
                "ssl_simulator.components.gvf",
                settings["gvf_traj"]["__class__"], **settings["gvf_traj"]["__params__"]
            )

        self.icons = None
        self.tails = None
        self.path_line = None
        self.quivers = None

        # Default visual properties
        kw_fig = dict(dpi=100, figsize=(6,9))
        kw_ax = dict(xlims=None, ylims=None)
        self.kw_patch = dict(fc="royalblue", ec="black", size=15, lw=1, zorder=3)

        # Update defaults with user-specified values
        self.kw_fig = parse_kwargs(kwargs, kw_fig)
        self.kw_ax = parse_kwargs(kwargs, kw_ax)
        self.kw_lines = dict(c="royalblue", lw=2)
        self.kw_field = None

        # Plotters
        self.gvf_plotter = PlotterGvfIk(self.gvf_traj, self.ax)
        self.robot_plotter = PlotterFixedwing(self.ax, self.data, **kwargs)

    def _config_axes(self):
        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$ [L]")
        self.ax.grid(True)
        self.ax.set_aspect("equal")
        config_axis(self.ax, **self.kw_ax)

    # ---------------------------------------------------------------------------------
        
    def draw(self, **kwargs):
        self._config_axes()
        
        # Extract derired data
        s = self.data["s"][-1]
        ke = self.data["ke"][-1]
        speed = self.data["speed"][-1,0]
        gamma = self.data["gamma"][-1,0]
        gamma_dot = self.data["gamma_dot"][-1,0]

        # ------------------------------------------------
        # Plot each robot's icon and tail
        self.robot_plotter.draw(**self.kw_lines)

        # Plot the GVF (traj + field)
        self.kw_field = dict(
            color="grey", alpha=0.8, zorder=1, lw=1.4, pts=30, pts_cond=200,
            s=s, ke=ke, gamma=gamma, gamma_dot=gamma_dot, speed=speed)
        self.kw_field = parse_kwargs(kwargs, self.kw_field)
 
        self.gvf_plotter.draw(**self.kw_field)

        # Plot the legend
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

        arr = self.ax.scatter([], [], c=self.kw_field["color"], alpha=self.kw_field["alpha"],
                                marker=r"$\uparrow$", s=60, label=r"$f(\phi(p),t_f)$")
        reg = mpatches.Patch(color='grey', alpha=0.2, label=r"$\{p \in \mathbb{R}^2 \; : \; \|\vartheta(p)\| < v\}$")
        self.ax.legend(handles=[arr, reg], fancybox=True, prop={"size": 16}, ncols=2, loc="upper left")

    def update(self):

        # Extract derired data
        speed = self.data["speed"][-1,0]
        gamma = self.data["gamma"][-1,0]
        gamma_dot = self.data["gamma_dot"][-1,0]

        # ------------------------------------------------
        # Update each robot's icon and tail
        self.robot_plotter.update()

        # Update GVF (traj + field)
        self.gvf_plotter.update(gamma=gamma, gamma_dot=gamma_dot, speed=speed)

#######################################################################################