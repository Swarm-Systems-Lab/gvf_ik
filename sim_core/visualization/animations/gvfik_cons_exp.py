"""
"""

__all__ = ["AnimationGvfIkConsExp"]

import numpy as np
from tqdm import tqdm

# Graphic tools
import matplotlib.pyplot as plt

# Animation tools
from matplotlib.animation import FuncAnimation

# Import tools from the Swarm Systems Lab Python Simulator
from ssl_simulator import load_pprz_data, parse_kwargs
from ssl_simulator import pprz_angle
from ssl_simulator.visualization import fixedwing_patch, config_data_axis
from ssl_simulator.visualization import smooth_interpolation

#######################################################################################

COLORS_PATCH = ["turquoise", "red"]
COLORS = ["turquoise", "red"]
COLORS_D = ["#20B2AA", "darkred"]

class AnimationGvfIkConsExp:
    def __init__(self, data1, data2, gps_img=None, idx_filt=None, debug=False, **kwargs):
        self.debug = debug
        self.gps_img = gps_img

        # Extract the data
        self.data_time = np.array(data1["Time"].to_list())

        self.data_x = np.array([data1["NAVIGATION:pos_y"].to_list(), 
                                data2["NAVIGATION:pos_y"].to_list()]).T
        self.data_y = np.array([data1["NAVIGATION:pos_x"].to_list(), 
                                data2["NAVIGATION:pos_x"].to_list()]).T
        self.data_theta = -pprz_angle(np.array([data1["ATTITUDE:psi"].to_list(), 
                                                data2["ATTITUDE:psi"].to_list()])).T + np.pi/2
        self.data_phi = np.array([data1["GVF:error"].to_list(), 
                                  data2["GVF:error"].to_list()]).T
        self.data_alt = np.array([data1["GPS:alt"].to_list(), 
                                  data2["GPS:alt"].to_list()]).T / 1000 - 720


        self.data_e = np.array([data1["GVF_IK_CONS:error"].to_list(), 
                                data2["GVF_IK_CONS:error"].to_list()]).T
        self.data_A = np.array([data1["GVF_IK:gamma_A"].to_list(), 
                                data2["GVF_IK:gamma_A"].to_list()]).T
        self.data_x_avg = np.array([data1["GVF_IK_CONS:x_avg"].to_list(), 
                                    data2["GVF_IK_CONS:x_avg"].to_list()]).T
        self.data_eik = np.array([data1["GVF_IK:error"].to_list(), 
                                  data2["GVF_IK:error"].to_list()]).T
        self.data_x_avg_dot_d = np.array([data1["GVF_IK_CONS:x_avg_dot_d"].to_list(), 
                                          data2["GVF_IK_CONS:x_avg_dot_d"].to_list()]).T

        self.data_speed = np.array([data1["GPS:speed"].to_list(), 
                                    data2["GPS:speed"].to_list()]).T / 100

        self.N = 2

        # self.data_omega_d = np.array(data1["GVF:omega_d"].to_list())
        # self.data_omega = np.array(data1["GVF:omega"].to_list())
        # self.omega = np.array(data1["GVF_IK:gamma_omega"].to_list())[0]

        # Filter telemetry data dropouts using smooth interpolation
        if idx_filt is not None:
            idx_start, idx_end = idx_filt

            # XY
            x_slice = -self.data_x[idx_start:idx_end, 0]  # Negate x values for interpolation
            y_slice = self.data_y[idx_start:idx_end, 0]

            smoothed_x, smoothed_y = smooth_interpolation(
                x_slice, y_slice, num_points=idx_end - idx_start
            )

            # Restore original x sign after interpolation
            self.data_x[idx_start:idx_end, 0] = -smoothed_x
            self.data_y[idx_start:idx_end, 0] = smoothed_y

            # # THETA
            # data_theta_inv = 2*np.pi + self.data_theta
            # self.data_theta = self.data_theta * (self.data_theta>=0) + \
            #                   data_theta_inv * (self.data_theta<0)
            
            # t_slice = self.data_time[idx_start:idx_end]
            # theta_slice = self.data_y[idx_start:idx_end, 0]

            # _, smoothed_theta = smooth_interpolation(
            #     t_slice, theta_slice, num_points=idx_end - idx_start
            # )

            # self.data_theta[idx_start:idx_end, 0] = smoothed_theta
            
        # -----------------------------------------------------------------------------

        kw_fig = {
            "dpi": 100,
            "figsize": (10,6)
        }

        kw_grid = {
            "hspace": 0,
            "wspace": 0.5
        }

        kw_ax = {
            "xlim":None, 
            "ylim":None
        }
        
        kw_patch = {
            # "fc": "None",
            # "ec": "red",
            "size": 50,
            "lw": 1,
            "zorder": 3,
        }

        kw_line = {
            "c": "k",
            "ls": "-",
            "lw": 1,
            "alpha": 0.6,
        }

        self.kw_fig = parse_kwargs(kwargs, kw_fig)
        self.kw_ax = parse_kwargs(kwargs, kw_ax)
        self.kw_grid = parse_kwargs(kwargs, kw_grid)
        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_line = parse_kwargs(kwargs, kw_line)

        # -----------------------------------------------------------------------------
        # Initialize the plot and axis configuration
        self.fig, self.ax = plt.subplots(**self.kw_fig)
        self.init_figure()

        # Init class variables
        self.data_time = self.data_time - self.data_time[0]


    def init_figure(self):

        # Axis configuration
        xlim, ylim = self.kw_ax["xlim"], self.kw_ax["ylim"]
        if xlim is not None:
            self.ax.set_xlim(ylim)
            self.ax.set_xlim(self.ax.get_xlim()[::-1])
        if ylim is not None:
            self.ax.set_ylim(xlim)

        self.ax.grid(True)
        self.ax.set_xlabel(r"$X$ [m]")
        self.ax.set_ylabel(r"$Y$  [m]")
        # ax.set_aspect("equal")
        config_data_axis(self.ax, x_step=100, y_step=100, y_right=False)

        self.ax.set_title(r"$k_A = 1.35$, $k_u = 0.06$, $w_\gamma = 0.6$rad/s")

        # -- Main Plot (XY)
        self.ax.plot([970, -670], [-411, -411], "lightgreen", zorder=2)
        self.ax.plot([970, -670], [-236, -236], "lightgreen", zorder=2)
        
        self.ax_patch = []
        self.ax_lines = []
        for i in range(self.N):
            line, = self.ax.plot(self.data_x[0,i], self.data_y[0,i], COLORS_PATCH[i], zorder=3)
            self.ax_lines.append(line)

            patch = fixedwing_patch(
                [self.data_x[0,i], self.data_y[0,i]], 
                self.data_theta[0,i], 
                fc=COLORS_PATCH[i], ec="k", **self.kw_patch)
            
            self.ax_patch.append(patch)
            self.ax.add_artist(patch)
        
        # Set the GPS background
        if self.gps_img is not None:
            if xlim is not None and ylim is not None:
                left, right = ylim
                bottom, top = xlim
                self.ax.imshow(self.gps_img, extent=(left, right, bottom, top))
            else:
                self.ax.imshow(self.gps_img)

        if self.debug:
            plt.show()

    def animate(self, iframe):
        if iframe < self.anim_frames_sim:
            for i in range(self.N):
                
                # Update trace
                self.ax_lines[i].set_data(self.data_x[0:iframe+1,i], self.data_y[0:iframe+1,i])

                # Update the icon
                self.ax_patch[i].remove()
                self.ax_patch[i] = fixedwing_patch(
                        [self.data_x[iframe,i], self.data_y[iframe,i]], 
                        self.data_theta[iframe,i], 
                        fc=COLORS_PATCH[i], ec="k", **self.kw_patch)
                self.ax_patch[i].set_zorder(10)
                self.ax.add_patch(self.ax_patch[i])
                
    def gen_animation(self, fps=None, anim_tf=None, wait_period=3):
        """
        Generate the animation object.
        """

        # Animation fps and frames
        if fps is None:
            dt = self.data_time[1] - self.data_time[0]
            self.fps = 1 / dt
        else:
            self.fps = fps

        self.anim_frames_sim = len(self.data_time)

        # Set wait period
        self.wait_its = int(wait_period * self.fps)
        self.anim_frames = self.anim_frames_sim + self.wait_its

        # Generate the animation
        print("Simulating {0:d} ({1:d}) frames... \nProgress:".format(
            self.anim_frames, self.anim_frames_sim))
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