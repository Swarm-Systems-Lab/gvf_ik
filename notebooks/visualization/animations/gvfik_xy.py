"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
from tqdm import tqdm

# Graphic tools
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# Animation tools
from matplotlib.animation import FuncAnimation

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import fixedwing_patch, config_data_axis
from ssl_simulator.gvf_trajectories import GvfEllipse

#######################################################################################

__all__ = ["AnimationXY"]

class AnimationXY:
    def __init__(
        self,
        gvf_traj,
        data,
        kw_alphainit=0.5,
        **kwargs
    ):
        # -----------------------------------------------------------------------------

        kw_fig = {
            "dpi": 100,
            "figsize": (6,6)
        }

        kw_ax = {
            "x_step": 50,
            "y_step": 50,
            "y_right": False,
            "xlims": None,
            "ylims": None  
        }
        
        kw_patch = {
            "fc": "None",
            "ec": "red",
            "size": 10,
            "lw": 1,
            "zorder": 3,
        }

        kw_line = {
            "c": "royalblue",
            "ls": "-",
            "lw": 2,
            "alpha": 0.7,
        }

        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_line = parse_kwargs(kwargs, kw_line)
        
        # -----------------------------------------------------------------------------
        self.gvf_traj = gvf_traj

        # Collect some data
        self.tdata = np.array(data["time"].tolist())
        self.xdata = np.array(data["p"].tolist())[:,0,0]
        self.ydata = np.array(data["p"].tolist())[:,0,1]
        self.thetadata = np.array(data["theta"].tolist())[:,0]

        # -----------------------------------------------------------------------------
        # Initialize the plot and axis configuration
        self.fig = plt.figure(**kw_fig)

        self.ax = self.fig.subplots()

        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$  [L]")
        self.ax.set_aspect("equal")

        config_data_axis(self.ax, **kw_ax)

        # -----------------------------------------------------------------------------
        # Draw the trajectory the level set
        self.gvf_traj.draw(self.fig, self.ax, lw=1.4, draw_field=False)

        # Initialize agent's icon
        icon_init = fixedwing_patch(
            [self.xdata[0], self.ydata[0]], self.thetadata[0], **self.kw_patch
        )
        self.agent_icon = fixedwing_patch(
            [self.xdata[0], self.ydata[0]], self.thetadata[0], **self.kw_patch
        )

        icon_init.set_alpha(kw_alphainit)
        self.agent_icon.set_zorder(10)

        self.ax.add_patch(icon_init)
        self.ax.add_patch(self.agent_icon)

        # Initialize agent's tail
        (self.agent_line,) = self.ax.plot(self.xdata[0], self.ydata[0], **kw_line)
        # -----------------------------------------------------------------------------

    def animate(self, i):
        # Update the icon
        self.agent_icon.remove()
        self.agent_icon = fixedwing_patch(
            [self.xdata[i], self.ydata[i]], self.thetadata[i], **self.kw_patch
        )
        self.agent_icon.set_zorder(10)
        self.ax.add_patch(self.agent_icon)

        # Update the tail
        if i > self.tail_frames:
            self.agent_line.set_data(
                self.xdata[i - self.tail_frames : i],
                self.ydata[i - self.tail_frames : i],
            )
        else:
            self.agent_line.set_data(self.xdata[0:i], self.ydata[0:i])

    def gen_animation(self, fps=None, anim_tf=None, tail_frames=500):
        """
        Generate the animation object.
        """

        # Animation fps and frames
        if anim_tf is None:
            anim_tf = self.tdata[-1]
        elif anim_tf > self.tdata[-1]:
            anim_tf = self.tdata[-1]

        dt = self.tdata[1] - self.tdata[0]
        if fps is None:
            self.fps = 1 / dt
        else:
            self.fps = fps
        self.anim_frames = int(anim_tf / dt)

        self.tail_frames = tail_frames

        # Generate the animation
        self.wait_state = 0
        self.waited_its = 0
        self.last_i = 0

        print("Simulating {0:d} frames... \nProgress:".format(self.anim_frames))
        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=tqdm(range(self.anim_frames), initial=1, position=0),
            interval=1 / self.fps * 1000,
        )
        anim.embed_limit = 40

        # Close plots and return the animation class to be compiled
        plt.close()
        return anim

#######################################################################################