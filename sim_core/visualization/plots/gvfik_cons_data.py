"""
"""

__all__ = ["PlotGvfIkConsData"]

import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Iterable
from ssl_simulator.visualization import fixedwing_patch
from ssl_simulator.components.gvf import GvfTrajectoryPlotter

A_FIT = 1.35

#######################################################################################

class PlotGvfIkConsData:
    def __init__(self, data, gvf_traj, **patch_kwargs):
        self.data = data
        self.gvf_traj = gvf_traj

        # Create subplots
        self.fig_gvf, self.ax_gvf = plt.subplots(figsize=(10,4))
        self.fig_data, self.ax_data = plt.subplots(2,3,figsize=(18,8))

        # Default patch properties
        self.path_kw = {
            "fc": "None",
            "ec": "red",
            "size": 15,
            "lw": 1,
            "zorder": 3,
        }

        # Update defaults with user-specified values
        self.path_kw.update(patch_kwargs)

    # ---------------------------------------------------------------------------------
    def config_axes(self):
        self.ax_gvf.set_xlabel(r"$X$ [m]")
        self.ax_gvf.set_ylabel(r"$Y$ [m]")
        self.ax_gvf.grid(True)
        self.ax_gvf.set_aspect("equal")

        for ax in self.ax_data.flatten():
            ax.grid(True)

    def plot(self, **line_kwargs):
        self.config_axes()
        
        # Extract derired data
        x = np.array(self.data["p"].tolist())[1:,:,0]
        y = np.array(self.data["p"].tolist())[1:,:,1]
        theta = np.array(self.data["theta"].tolist())

        N = x.shape[1]

        # ------------------------------------------------
        # Plot the robots
        self.ax_gvf.plot(x, y, "b", **line_kwargs)

        for i in range(N):
            patch_init = fixedwing_patch([x[0,i], y[0,i]], theta[0,i], **self.path_kw)
            patch_final = fixedwing_patch([x[-1,i], y[-1,i]], theta[-1,i], **self.path_kw)

            self.ax_gvf.add_artist(patch_init)
            self.ax_gvf.add_artist(patch_final)

        # Plot the GVF
        if isinstance(self.gvf_traj, Iterable):
            for i in range(len(self.gvf_traj)):
                gvf_traj_plotter = GvfTrajectoryPlotter(self.gvf_traj[i], self.fig_gvf, self.ax_gvf)
                gvf_traj_plotter.draw(lw=1.4, draw_field=False)
        else:   
                gvf_traj_plotter = GvfTrajectoryPlotter(self.gvf_traj, self.fig_gvf, self.ax_gvf)
                gvf_traj_plotter.draw(lw=1.4, draw_field=False)

        return self.ax_gvf
        
        # ------------------------------------------------
    
    def plot_data(self, **line_kwargs):
        self.config_axes()

        # Extract derired data
        time_values = np.array(self.data["time"].tolist())[1:]
        theta_dot = np.array(self.data["theta_dot"].tolist())[1:,:]
        speed = np.array(self.data["speed"].tolist())[0,:]

        gvf_ke = np.array(self.data["ke"].tolist())[0]
        gvf_kn = np.array(self.data["kn"].tolist())[0]
        gvf_ka = np.array(self.data["ka"].tolist())[0]
        
        gamma_A = np.array(self.data["gamma_A"].tolist())[1:]
        gamma_omega = np.array(self.data["gamma_omega"].tolist())[0]
        e = np.array(self.data["e_cons"].tolist())[1:,:]
        x_traj = np.array(self.data["x_traj"].tolist())[1:,:]

        # Calculate \dot{\bar x} during the first period
        period = 2 * np.pi / gamma_omega

        t0_tangent = time_values[(gamma_A != 0).any(axis=1)][0]

        x_traj_avg = np.zeros_like(x_traj)
        for i,t in enumerate(time_values):
            if t > period:
                mask = np.logical_and((t - period) < time_values, time_values < t)
            else:
                mask = time_values <= t
            n_values = np.sum(mask)
            x_traj_avg[i] = np.sum(x_traj[mask,:], axis=0) / n_values

        x_traj_dot = np.zeros_like(x_traj)
        for i, t in enumerate(time_values):
            if i > 0 and t >= period:
                x_traj_dot[i-1] = (x_traj_avg[i] - x_traj_avg[i-1]) \
                                / (time_values[i] - time_values[i-1])
                x_traj_dot[i] = x_traj_dot[i-1]

        # ------------------------------------------------
        # Print controller parameters 
        print(f"ke = {gvf_ke:.2f}   kn = {gvf_kn:.2f}   ka = {gvf_ka:.2f}")

        # Plot control-related variables
        dashed_kwargs = {"color":"k", "lw":2, "ls":"--", "zorder":3}
        for ax in self.ax_data.flatten():
            ax.axvline(t0_tangent, **dashed_kwargs)

        self.ax_data[0,0].set_ylabel(r"$x$")
        self.ax_data[0,0].plot(time_values, x_traj, "b", **line_kwargs)

        self.ax_data[0,1].set_ylabel(r"$A$")
        self.ax_data[0,1].plot(time_values, gamma_A, "b", **line_kwargs)

        self.ax_data[0,2].set_ylabel(r"$\omega$")
        self.ax_data[0,2].plot(time_values, theta_dot, "b", **line_kwargs)
        
        # self.ax_data[1,0].set_ylabel(r"$\bar x$")
        # self.ax_data[1,0].plot(time_values, x_traj_avg, "b", **line_kwargs)

        self.ax_data[1,0].set_ylabel(r"$\dot{\bar x}$")
        self.ax_data[1,0].plot(time_values, x_traj_dot, "b", **line_kwargs)
        epsilon = np.sqrt(A_FIT**2 - 1)/A_FIT * speed[0]
        self.ax_data[1,0].axhline(epsilon, **dashed_kwargs)
        self.ax_data[1,0].text(120, epsilon*0.9, r"$\epsilon$")

        self.ax_data[1,1].set_ylabel(r"$e$")
        self.ax_data[1,1].plot(time_values, e, "b", **line_kwargs)


        # ------------------------------------------------

    def save(self, filename, dpi=100):
        plt.savefig(filename, dpi=dpi)

#######################################################################################