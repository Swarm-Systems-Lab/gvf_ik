"""
"""

__all__ = ["AnimationOscInst"]

import numpy as np
from tqdm import tqdm

# Graphic tools
import matplotlib.pyplot as plt

# Animation tools
from matplotlib.animation import FuncAnimation

# Import tools from the Swarm Systems Lab Python Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import config_data_axis

#######################################################################################

class AnimationOscInst:
    def __init__(self, data, debug=False, **kwargs):
        self.data = data
        self.debug = debug

        # -----------------------------------------------------------------------------
        # Collect some data
        self.tdata = np.array(data["time"].tolist())
        self.xdata = np.array(data["p"].tolist())[:,:,0]
        self.ydata = np.array(data["p"].tolist())[:,:,1]
        
        self.speed = np.array(self.data["speed"].tolist())[0,0]
        self.omega = np.array(self.data["gamma_omega"].tolist())[0,0]
        self.A = np.array(self.data["gamma_A"].tolist())[0,:]
        self.x_dot = np.array(self.data["x_dot"].tolist())

        self.N = len(self.A)

        if self.debug:
            print(self.ydata[-1,:])

        # -----------------------------------------------------------------------------

        kw_fig = {
            "dpi": 100,
            "figsize": (10,6)
        }

        kw_grid = {
            "hspace": 0,
            "wspace": 0.3
        }

        kw_patch = {
            "fc": "None",
            "ec": "red",
            # "size": 10,
            # "lw": 1,
            # "zorder": 3,
        }

        kw_line = {
            "c": "k",
            "ls": "-",
            # "lw": 2,
            "alpha": 0.6,
        }

        self.kw_fig = parse_kwargs(kwargs, kw_fig)
        self.kw_grid = parse_kwargs(kwargs, kw_grid)
        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_line = parse_kwargs(kwargs, kw_line)

        # -----------------------------------------------------------------------------
        # Initialize the plot and axis configuration
        self.fig = plt.figure(**self.kw_fig)
        grid = plt.GridSpec(1, 2, **self.kw_grid)
        self.ax1 = self.fig.add_subplot(grid[:, 0])
        self.ax2 = self.fig.add_subplot(grid[:, 1])
        
        self.init_figure()

    def config_axes(self):
        self.ax1.set_xlabel(r"$X$ [m]")
        self.ax1.set_ylabel(r"$Y$ [m]")
        self.ax1.grid(True)

        self.ax2.set_xlabel(r"$t$ [s]")
        self.ax2.set_ylabel(r"$\dot{x}$ [m/s]")
        self.ax2.grid(True)

        config_data_axis(self.ax1, y_right=False, x_step=50, y_step=5)
        config_data_axis(self.ax2, y_right=False, 
                         ylims=[0,self.speed+2],
                         xlims=[0,self.tdata[-1]],
                         x_step=5, y_step=2)

    def init_figure(self):
        # Configure axes for plotting
        self.config_axes()
        self.ax1.plot(self.xdata, self.ydata, "b", alpha=0)

        # Plot
        self.ax1.set_title(rf"$v = {self.speed:.0f}$m/s, $w_\gamma = {self.omega:.1f}$rad/s, $A = {self.A[0]:.1f}$m")
        self.ax2.set_title(rf"$t = {0:.2f}$s")

        self.ax1_dots = []
        self.ax1_lines = []
        self.ax2_lines = []
        for i in range(self.N):
            kw_point = {"alpha":1, "zorder":3}
            kw_point.update(self.kw_patch)

            ax1_dot = self.ax1.scatter(self.xdata[0,i], self.ydata[0,i], marker="o", **kw_point)
            ax1_line, = self.ax1.plot(self.xdata[0,i], self.ydata[0,i], **self.kw_line)
            ax2_line, = self.ax2.plot(self.tdata[0], self.x_dot[0,i], "k", lw=1.5)

            self.ax1_dots.append(ax1_dot)
            self.ax1_lines.append(ax1_line)
            self.ax2_lines.append(ax2_line)

        if self.debug:
            plt.show()

    def animate(self, iframe):
        if iframe < self.anim_frames_sim:
            for i in range(self.N):

                j = iframe

                self.ax2.set_title(rf"$t = {self.tdata_anim[j]:.2f}$s")

                # First axis
                self.ax1_dots[i].set_alpha(1)
                self.ax1_dots[i].set_offsets([self.xdata_anim[j,i], self.ydata_anim[j,i]])
                self.ax1_lines[i].set_data(self.xdata_anim[0:j+1,i], self.ydata_anim[0:j+1,i])

                # Second axis
                self.ax2_lines[i].set_data(self.tdata_anim[0:j+1], self.x_dot_anim[0:j+1,i])

    
    def gen_animation(self, fps=None, wait_period=3, factor=1):
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
        self.tdata_anim = self.tdata[0:len(self.tdata):factor]
        self.xdata_anim = self.xdata[0:len(self.tdata):factor,:]
        self.ydata_anim = self.ydata[0:len(self.tdata):factor,:]
        self.x_dot_anim = self.x_dot[0:len(self.tdata):factor,:]
        self.A_anim = self.A

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