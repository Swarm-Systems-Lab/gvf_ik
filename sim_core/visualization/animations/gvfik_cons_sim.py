"""
"""

__all__ = ["AnimationGvfIkConsInterpSim"]

import numpy as np
from tqdm import tqdm
from collections.abc import Iterable

# Graphic tools
import matplotlib.pyplot as plt

# Animation tools
from matplotlib.animation import FuncAnimation

# Import tools from the Swarm Systems Lab Python Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import fixedwing_patch, config_data_axis
from ssl_simulator.components.gvf import GvfTrajectoryPlotter

#######################################################################################

class AnimationGvfIkConsInterpSim:
    def __init__(self, data, gvf_traj, debug=False, **kwargs):
        self.data = data
        self.gvf_traj = gvf_traj
        self.debug = debug

        # -----------------------------------------------------------------------------
        # Collect some data
        self.tdata = np.array(data["time"].tolist())
        self.xdata = np.array(data["p"].tolist())[:,:,0]
        self.ydata = np.array(data["p"].tolist())[:,:,1]
        self.theta_data = np.array(self.data["theta"].tolist())

        self.gvf_s = np.array(self.data["s"].tolist())[0]
        self.gvf_ke = np.array(self.data["ke"].tolist())[0]
        self.Z = np.array(self.data["Z"].tolist())[0,:,:]

        self.N = self.xdata.shape[1]

        if self.debug:
            print(self.ydata[-1,:])

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
            "size": 15,
            "lw": 1,
            "zorder": 3,
        }

        kw_line = {
            "c": "b",
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
        self.ax.set_xlabel(r"$X$ [m]")
        self.ax.set_ylabel(r"$Y$ [m]")
        self.ax.set_aspect("equal")
        config_data_axis(self.ax, **self.kw_ax)

    def init_figure(self):
        # Configure axes for plotting
        self.config_axes()
        self.ax.set_title(r"$k_A = 1.35$, $k_u = 0.16$, $w_\gamma = 0.6$rad/s")

        self.ax_lines = []
        self.ax_patch = []
        for i in range(self.N):
            line, = self.ax.plot(self.xdata[0,i], self.ydata[0,i], **self.kw_line)
            self.ax_lines.append(line)

            patch = fixedwing_patch(
                [self.xdata[0,i], self.ydata[0,i]], self.theta_data[0,i], 
                **self.kw_patch)
            
            self.ax_patch.append(patch)
            self.ax.add_artist(patch)

        # Plot the graph
        self.ax_edge_lines = []
        for edge in self.Z:
            i,j = edge
            ax_edge_line, = self.ax.plot([self.xdata[0,i], self.xdata[0,j]], 
                                        [self.ydata[0,i], self.ydata[0,j]], "--", c="grey")
            self.ax_edge_lines.append(ax_edge_line)

        # Plot the GVF
        if isinstance(self.gvf_traj, Iterable):
            for i in range(len(self.gvf_traj)):
                gvf_traj_plotter = GvfTrajectoryPlotter(self.gvf_traj[i], self.fig, self.ax)
                gvf_traj_plotter.draw(lw=1.4, draw_field=False)
        else:   
                gvf_traj_plotter = GvfTrajectoryPlotter(self.gvf_traj, self.fig, self.ax)
                gvf_traj_plotter.draw(lw=1.4, draw_field=False)

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

                self.ax_patch[i] = fixedwing_patch(
                    [self.xdata_anim[iframe,i], self.ydata_anim[iframe,i]], 
                    self.theta_data_anim[iframe,i], 
                    **self.kw_patch)
                
                self.ax_patch[i].set_zorder(10)
                self.ax.add_patch(self.ax_patch[i])

                # Update graph lines
                for idx, edge in enumerate(self.Z):
                    i,j = edge
                    self.ax_edge_lines[idx].set_data(
                        [self.xdata_anim[iframe,i], self.xdata_anim[iframe,j]], 
                        [self.ydata_anim[iframe,i], self.ydata_anim[iframe,j]])
    
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