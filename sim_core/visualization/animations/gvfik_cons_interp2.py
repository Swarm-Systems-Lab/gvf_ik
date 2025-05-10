"""
"""

__all__ = ["AnimationGvfIkConsInterp2"]

import numpy as np
from tqdm import tqdm

# Graphic tools
import matplotlib.pyplot as plt

# Animation tools
from matplotlib.animation import FuncAnimation

# Import tools from the Swarm Systems Lab Python Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import config_data_axis

A_FIT = 1.35

#######################################################################################

class AnimationGvfIkConsInterp2:
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
        self.x_dot_avg = np.array(self.data["x_dot_avg"].tolist())

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
            "wspace": 0.5
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
        grid = plt.GridSpec(1, 3, **self.kw_grid)
        self.ax1 = self.fig.add_subplot(grid[:, 0:2])
        self.ax2 = self.fig.add_subplot(grid[:, 2])
        
        self.init_figure()

    def config_axes(self, y):
        self.ax1.set_xlabel(r"$X$ [m]")
        self.ax1.set_ylabel(r"$Y$ [m]")
        self.ax1.grid(True)

        self.ax2.set_xlabel(r"$\dot{\bar x}$ [m/s]")
        self.ax2.set_ylabel(r"$A$ [m]")
        self.ax2.grid(True)

        # Compute x/y-axis limits with padding
        y_min, y_max = np.min(y), np.max(y)
        pad_dist = abs(y_max - y_min) * 0.1
        y_lower_pad, y_upper_pad = y_min - pad_dist, y_max + 2*pad_dist

        config_data_axis(self.ax1, y_right=False, x_step=20, y_step=5)
        config_data_axis(self.ax2, y_right=False, ylims=[y_lower_pad, y_upper_pad],
                         x_step=2, y_step=2)

        return y_lower_pad, y_upper_pad

    def init_figure(self):
        # Configure axes for plotting
        self.y_lower_pad, _ = self.config_axes(self.A)
        self.ax1.plot(self.xdata, self.ydata, "b", alpha=0)

        # Fitting line
        x_lin = np.linspace(np.min(self.x_dot_avg[-1,:]), np.max(self.x_dot_avg[-1,:]), 100)
        A_lin = A_FIT * np.sqrt(self.speed**2 - x_lin**2) / self.omega
        self.ax2.plot(x_lin, A_lin, "k", 
                      label=r"$k_A \sqrt{v^2 - \dot{\bar x}^2} / w_\gamma$")
        
        self.ax2.legend(loc="upper right", fontsize=12)

        self.ax1.set_title(rf"$v = {self.speed:.0f}$m/s, $k_A = 1.35$, $w_\gamma = {self.omega:.1f}$rad/s")
        self.ax2.set_title(rf"$t = {0:.2f}$s")

        self.ax1_lines = []
        self.ax1_squares = []
        self.ax2_squares = []
        self.ax1_labels = []
        self.ax2_labels = []
        for i in range(self.N):
            kw_point = {"alpha":1, "zorder":3, "fc":"None", "ec":"red", "marker":"s", "s":30}
            kw_text = {"fontdict": {"size":9}, "zorder":4}

            ax1_line, = self.ax1.plot(self.xdata[0,i], self.ydata[0,i], **self.kw_line)
            ax1_square = self.ax1.scatter(self.xdata[0,i], self.ydata[0,i], **kw_point)
            ax2_square = self.ax2.scatter(self.x_dot_avg[0,i], self.A[i], **kw_point)

            ax1_label = self.ax1.text(self.xdata[0,i]-1, self.ydata[0,i]+1, str(i+1), **kw_text)
            ax2_label = self.ax2.text(self.x_dot_avg[0,i]-1.5, self.A[i]-0.2, str(i+1), **kw_text)
            
            self.ax1_lines.append(ax1_line)
            self.ax1_squares.append(ax1_square)
            self.ax2_squares.append(ax2_square)
            self.ax1_labels.append(ax1_label)
            self.ax2_labels.append(ax2_label)

        if self.debug:
            plt.show()

    def animate(self, iframe):
        if iframe < self.anim_frames:
            for i in range(self.N):

                if self.launch:
                    if (iframe >= self.anim_frames_sim * i) and (iframe < self.anim_frames_sim * (i+1)):
                        j = iframe - self.anim_frames_sim * i
                    else:
                        continue
                else:
                    j = iframe

                self.ax2.set_title(rf"$t = {self.tdata_anim[j]:.2f}$s")

                # First axis
                self.ax1_squares[i].set_alpha(1)
                self.ax1_squares[i].set_offsets((self.xdata_anim[j,i], self.ydata_anim[j,i]))
                self.ax1_lines[i].set_data(self.xdata_anim[0:j+1,i], self.ydata_anim[0:j+1,i])
                self.ax1_labels[i].set_position((self.xdata_anim[j,i]-1.2, self.ydata_anim[j,i]+1))

                # Second axis
                self.ax2_squares[i].set_alpha(1)
                self.ax2_squares[i].set_offsets((self.x_dot_avg_anim[j,i], self.A_anim[i]))
                self.ax2_labels[i].set_position((self.x_dot_avg_anim[j,i]-1.5, self.A_anim[i]-0.2))

    
    def gen_animation(self, fps=None, wait_period=3, factor=1, launch=False):
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
        self.x_dot_avg_anim = self.x_dot_avg[0:len(self.tdata):factor,:]
        self.A_anim = self.A

        # Launch one by one
        self.launch = launch
        if launch:
            self.anim_frames = self.anim_frames_sim * self.N
        else:
            self.anim_frames = self.anim_frames_sim

        # Set wait period
        self.wait_its = int(wait_period * self.fps)
        self.anim_frames_wait = self.anim_frames + self.wait_its

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