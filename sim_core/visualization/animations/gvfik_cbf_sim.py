"""
"""

__all__ = ["AnimationGvfIkCBFSim"]

import numpy as np
from tqdm import tqdm
from collections.abc import Iterable

# Graphic tools
import matplotlib.pyplot as plt

# Animation tools
from matplotlib.animation import FuncAnimation

# Import tools from the Swarm Systems Lab Python Simulator
from ssl_simulator import parse_kwargs, load_class
from ssl_simulator.visualization import fixedwing_patch, config_data_axis
from ssl_simulator.components.gvf import PlotterGvf

#######################################################################################

class AnimationGvfIkCBFSim:
    def __init__(self, data, settings, debug=False, **kwargs):
        self.data = data
        self.gvf_traj = load_class(
            "ssl_simulator.components.gvf",
            settings["gvf_traj"]["__class__"], **settings["gvf_traj"]["__params__"]
        )
        self.kw_ax = kwargs
        self.debug = debug

        # -----------------------------------------------------------------------------
        # Collect some data
        self.tdata = np.array(data["time"].tolist())
        self.xdata = np.array(data["p"].tolist())[:,:,0]
        self.ydata = np.array(data["p"].tolist())[:,:,1]
        self.theta_data = np.array(self.data["theta"].tolist())
        self.speed = np.array(self.data["speed"].tolist())

        self.obstacles = settings["obstacles"]
        self.col_rad = settings["col_rad"]
        self.gvf_s = settings["s"]
        self.gvf_ke = settings["ke"]

        self.N = self.xdata.shape[1]

        # -----------------------------------------------------------------------------

        kw_fig = {
            "dpi": 100,
            "figsize": (10,6)
        }

        kw_ax = {
            "x_step": 100,
            "y_step": 50,
            "y_right": False,
            "xlims": [-20,1300],
            "ylims": [-55,215] 
        }
        
        kw_patch = {
            "fc": "None",
            "ec": "red",
            "size": 12,
            "lw": 1,
            "zorder": 3,
        }

        kw_line = {
            "c": "red",
            "ls": "-",
            "lw": 1.2,
            "alpha": 0.7,
        }

        self.kw_fig = parse_kwargs(kwargs, kw_fig)
        self.kw_ax = parse_kwargs(kwargs, kw_ax)
        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_line = parse_kwargs(kwargs, kw_line)

        # -----------------------------------------------------------------------------
        # Initialize the plot and axis configuration
        self.fig, self.ax = plt.subplots(**self.kw_fig)
        self.init_figure()

    def config_axes(self):
        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$ [L]")
        self.ax.set_aspect("equal")
        config_data_axis(self.ax, **self.kw_ax)

    def init_figure(self):
        # Configure axes for plotting
        self.config_axes()

        self.ax_lines = []
        self.ax_patch = []
        self.ax_patch_coll = []
        for i in range(self.N):
            if i not in self.obstacles:
                line, = self.ax.plot(self.xdata[0,i], self.ydata[0,i], **self.kw_line)
                patch = fixedwing_patch(
                    [self.xdata[0,i], self.ydata[0,i]], self.theta_data[0,i], 
                    **self.kw_patch)
                patch_coll = plt.Circle((0,0),0) # Dummy
            else:
                kw_line = parse_kwargs(dict(c="grey"), self.kw_line)
                line, = self.ax.plot(self.xdata[0,i], self.ydata[0,i], **kw_line)
                patch = plt.Circle(
                    (self.xdata[0,i], self.ydata[0,i]), radius=self.col_rad/3, 
                    fc="lightgrey", ec="black", lw=1, zorder=1)
                patch_coll = plt.Circle(
                    (self.xdata[0,i], self.ydata[0,i]), radius=self.col_rad, 
                    fc="None", ec="black", lw=1, ls="--", zorder=1)
            
            self.ax_lines.append(line)
            self.ax_patch.append(patch)
            self.ax_patch_coll.append(patch_coll)
            self.ax.add_artist(patch)
            self.ax.add_artist(patch_coll)

        # Plot the GVF
        self.kw_field = dict(
            color="grey", alpha=0.5, zorder=1, lw=2, pts=30,
            s=self.gvf_s, ke=self.gvf_ke, gamma=0, gamma_dot=0, speed=self.speed[0])

        gvf_traj_plotter = PlotterGvf(self.gvf_traj, self.ax)
        gvf_traj_plotter.draw(**self.kw_field)

        if self.debug:
            plt.show()

    def animate(self, iframe):
        if iframe < self.anim_frames_sim:
            for i in range(self.N):
                # Update traces
                self.ax_lines[i].set_data(self.xdata_anim[0:iframe+1,i], 
                                          self.ydata_anim[0:iframe+1,i])

                # Update the icon
                self.ax_patch[i].remove()
                self.ax_patch_coll[i].remove()

                if i not in self.obstacles:
                    self.ax_patch[i] = fixedwing_patch(
                        [self.xdata_anim[iframe,i], self.ydata_anim[iframe,i]], 
                        self.theta_data_anim[iframe,i], 
                        **self.kw_patch)
                    self.ax_patch_coll[i] = plt.Circle((0,0),0) # Dummy
                else:
                    self.ax_patch[i] = plt.Circle(
                        (self.xdata_anim[iframe,i], self.ydata_anim[iframe,i]), 
                        radius=self.col_rad/3, 
                        fc="lightgrey", ec="black", lw=1, zorder=1)
                    self.ax_patch_coll[i] = plt.Circle(
                        (self.xdata_anim[iframe,i], self.ydata_anim[iframe,i]), radius=self.col_rad, 
                        fc="None", ec="black", lw=1, ls="--", zorder=1)
                
                self.ax_patch[i].set_zorder(10)
                self.ax_patch_coll[i].set_zorder(10)
                self.ax.add_patch(self.ax_patch[i])
                self.ax.add_patch(self.ax_patch_coll[i])
    
    def gen_animation(self, fps=None, factor=1, wait_period=3):
        """
        Generate the animation object.
        """
        # Animation fps and frames
        if fps is None:
            dt = self.tdata[1] - self.tdata[0]
            self.fps = 1 / dt
        else:
            self.fps = fps
        
        self.anim_frames_sim = len(self.tdata) // factor

        # Animation data
        self.xdata_anim = self.xdata[0:len(self.tdata):factor,:]
        self.ydata_anim = self.ydata[0:len(self.tdata):factor,:]
        self.theta_data_anim = self.theta_data[0:len(self.tdata):factor,:]

        if (self.anim_frames_sim < len(self.tdata) / factor):
            print("Warning: The choosen factor is probably wrong!")

        # Set wait period
        self.wait_its = int(wait_period * self.fps)
        self.anim_frames_wait = self.anim_frames_sim + self.wait_its

        # Generate the animation
        print("Simulating {0:d} ({1:d}) frames... \nProgress:".format(
            self.anim_frames_wait, self.anim_frames_sim))
        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=tqdm(range(self.anim_frames_wait), initial=1, position=0),
            interval=1 / self.fps * 1000,
        )
        anim.embed_limit = 40

        # Close plots and return the animation class to be compiled
        plt.close()
        return anim

#######################################################################################