"""
"""

__all__ = ["PlotGvfIkConsExp"]

import numpy as np

# Graphic tools
import matplotlib.pyplot as plt

# Import tools from the Swarm Systems Lab Python Simulator
from ssl_simulator.math import pprz_angle
from ssl_simulator.visualization import fixedwing_patch, config_data_axis
from ssl_simulator.visualization import smooth_interpolation

#######################################################################################

COLORS_PATCH = ["turquoise", "red"]
COLORS = ["turquoise", "red"]
COLORS_D = ["#20B2AA", "darkred"]

class PlotGvfIkConsExp:
    def __init__(self, data1, data2, **kw_patch):
        
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

        # self.data_omega_d = np.array(data1["GVF:omega_d"].to_list())
        # self.data_omega = np.array(data1["GVF:omega"].to_list())
        # self.omega = np.array(data1["GVF_IK:gamma_omega"].to_list())[0]

        # Calculate the actual x_avg_dot
        self.data_x_avg_dot = np.zeros_like(self.data_x_avg_dot_d)
        for i in range(2):
            for j in range(len(self.data_time)):
                n = 10
                if j > n-1:
                    self.data_x_avg_dot[j-n,i] = (self.data_x_avg[j,i] - self.data_x_avg[j-n,i]) \
                                    / (self.data_time[j] - self.data_time[j-n])
                    self.data_x_avg_dot[j,i] = self.data_x_avg_dot[j-n,i]

        # Init class variables
        self.data_time = self.data_time - self.data_time[0]

        # Default patch properties
        self.path_kw = {

            "size": 50,
            "lw": 1,
            "zorder": 3,
        }

        # Update defaults with user-specified values
        self.path_kw.update(kw_patch)

    def plot(self, backgorund_img=None, xlim=None, ylim=None, 
             idx_list=[], idx_filt=None, **kw_fig):
        
        # Initialize the figure
        fig = plt.figure(**kw_fig)
        grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.3)
        ax = fig.add_subplot(grid[0:2, :])
        axr1 = fig.add_subplot(grid[2, 0])
        axr2 = fig.add_subplot(grid[2, 1])
        axr3 = fig.add_subplot(grid[2, 2])
        axr4 = fig.add_subplot(grid[2, 3])
        # axr5 = fig.add_subplot(grid[2, 4])

        # Axis configuration
        if xlim is not None:
            ax.set_xlim(ylim)
            ax.set_xlim(ax.get_xlim()[::-1])
        if ylim is not None:
            ax.set_ylim(xlim)

        ax.set_xlabel(r"$X$ [m]")
        ax.set_ylabel(r"$Y$  [m]")
        # ax.set_aspect("equal")
        config_data_axis(ax, x_step=100, y_step=100, y_right=False)

        # Right axis configuration
        axr1.set_ylabel(r"$\bar x_j - \bar x_i$ [m]")
        axr1.set_xlabel(r"$t$ [s]")
        config_data_axis(axr1, x_step=20, y_step=50, y_right=False)

        axr2.set_ylabel(r"$\dot{\bar{x}}_{i}$ [m]")
        axr2.set_xlabel(r"$t$ [s]")
        config_data_axis(axr2, x_step=20, y_step=4, y_right=False)

        axr3.set_ylabel(r"$A_{i,d}$ [m]")
        axr3.set_xlabel(r"$t$ [s]")
        config_data_axis(axr3, x_step=20, y_step=4, y_right=False)

        axr4.set_ylabel(r"$v_i$ [m/s]")
        axr4.set_xlabel(r"$t$ [s]")
        config_data_axis(axr4, x_step=20, y_step=1, y_right=False, ylims=[12,18])

        # axr5.set_ylabel(r"AGL [m]")
        # axr5.set_xlabel(r"$t$ [s]")
        # config_data_axis(axr5, x_step=20, y_step=5, y_right=False, ylims=[40,60])
        

        # -- Main Plot (XY)
        
        # Filter telemetry data dropouts using smooth interpolation
        if idx_filt is not None:
            idx_start, idx_end = idx_filt
            x_slice = -self.data_x[idx_start:idx_end, 0]  # Negate x values for interpolation
            y_slice = self.data_y[idx_start:idx_end, 0]

            smoothed_x, smoothed_y = smooth_interpolation(
                x_slice, y_slice, num_points=idx_end - idx_start
            )

            # Restore original x sign after interpolation
            self.data_x[idx_start:idx_end, 0] = -smoothed_x
            self.data_y[idx_start:idx_end, 0] = smoothed_y

        ax.plot(self.data_x[:idx_list[-1],0], self.data_y[:idx_list[-1],0], COLORS_PATCH[0], zorder=3)
        ax.plot(self.data_x[:idx_list[-1],1], self.data_y[:idx_list[-1],1], COLORS_PATCH[1], zorder=3)
        ax.plot([970, -670], [-411, -411], "lightgreen", zorder=2)
        ax.plot([970, -670], [-236, -236], "lightgreen", zorder=2)
        
        for idx in idx_list:

            for i in range(2):
                ax.add_artist(fixedwing_patch(
                    [self.data_x[idx,i], self.data_y[idx,i]], self.data_theta[idx,i], 
                    fc=COLORS_PATCH[i], ec="k", **self.path_kw))

            ax.text(self.data_x[idx,0]+5, -330,
                    rf"${self.data_time[idx]:.0f}$s".format(), c="white", 
                    fontdict={"size":20, "weight":"bold"})

        if backgorund_img is not None:
            if xlim is not None and ylim is not None:
                left, right = ylim
                bottom, top = xlim
                ax.imshow(backgorund_img, extent=(left, right, bottom, top))
            else:
                ax.imshow(backgorund_img)

        # -- Data plot 1
        axr1.axhline(0, color="k", ls="-", lw=1)
        for i in [1,0]:
            #axr1.plot(self.data_time[:idx_list[-1]], self.data_phi[:idx_list[-1],i], lw=1.5, color=COLORS[i])
            axr1.plot(self.data_time[:idx_list[-1]], self.data_e[:idx_list[-1],i], lw=1.5, color=COLORS[i], alpha=0.9)

        # -- Data plot 2
        axr2.axhline(0, color="k", ls="-", lw=1)
        lines = []
        for i in [1,0]:
            axr2.plot(self.data_time[:idx_list[-1]], self.data_x_avg_dot[:idx_list[-1],i], COLORS[i], lw=1.5, alpha=0.9)
            line, = axr2.plot(self.data_time[:idx_list[-1]], self.data_x_avg_dot_d[:idx_list[-1],i], "--", c=COLORS_D[i], lw=1.5)
            lines.append(line)
        axr2.legend(lines, ["",r"$\dot{\bar{x}}_{i,d}$"], ncol=2, columnspacing=0.1, fontsize=12)

        # -- Data plot 3
        axr3.axhline(0, color="k", ls="-", lw=1)
        for i in [1,0]:
            axr3.plot(self.data_time[:idx_list[-1]], self.data_A[:idx_list[-1],i], COLORS[i], lw=1.5, alpha=0.9)

        # -- Data plot 4
        axr4.axhline(0, color="k", ls="-", lw=1)
        axr4.axhline(16, color="k", ls="--", lw=1)
        for i in [1,0]:
            axr4.plot(self.data_time[:idx_list[-1]], self.data_speed[:idx_list[-1],i], COLORS[i], lw=1.5, alpha=0.9)

        # -- Data plot 5
        # axr5.axhline(0, color="k", ls="-", lw=1)
        # # axr5.axhline(16, color="k", ls="--", lw=1)
        # for i in [1,0]:
        #     axr5.plot(self.data_time[:idx_list[-1]], self.data_alt[:idx_list[-1],i], COLORS[i], lw=1.5, alpha=0.9)

        # ax_inset = inset_axes(axr3, width="40%", height="40%", loc="lower right")
        # config_data_axis(ax_inset, x_step=10, y_step=5, y_right=False)
        # ax_inset.set_xlim(80, 100)
        # ax_inset.set_ylim(-10, 10) 
        # mark_inset(axr1, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
        
        # -> Show the plot <-
        ax.grid(True)
        plt.show()

#######################################################################################