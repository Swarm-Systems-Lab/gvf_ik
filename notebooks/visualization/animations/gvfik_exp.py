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
from ssl_simulator.math import pprz_angle
from ssl_simulator.visualization import fixedwing_patch
from ssl_simulator.gvf_trajectories import GvfEllipse

__all__ = ["AnimationTelemetry"]

#######################################################################################

class AnimationTelemetry:
    def __init__(
        self,
        data,
        sim_kw_args: dict,
        A=0,
        omega=0,
        t0_sin=0,
        dpi=100,
        figsize=(6, 6),
        xlims=None,
        ylims=None,
        ytick_sep_phi=0.5,
        ytick_sep_roll=0.2,
        anim_tf=None,
        fps=None,
        kw_color="royalblue",
        kw_sz=7,
        kw_lw=0.5,
        kw_alphainit=0.5,
        tail_lw=1,
        wait_period=0,
    ):

        self.gvf_traj = sim_kw_args["gvf_traj"]

        # Collect some data
        self.tdata = np.array(data["Time"].to_list())
        self.tdata -= self.tdata[0]
        self.xdata = np.array(data["NAVIGATION:pos_x"].to_list())
        self.ydata = np.array(data["NAVIGATION:pos_y"].to_list())
        self.gps_speed = np.array(data["GPS:speed"].to_list()) / 100

        self.gvf_phi = np.array(data["GVF:phi"].to_list())
        self.ke = np.array(data["GVF:ke"].to_list())[0]
        self.gvf_tnorm = np.array(data["GVF:t_norm"].to_list())

        self.att_roll = np.array(data["ATTITUDE:phi"].to_list())
        self.att_yaw = pprz_angle(np.array(data["ATTITUDE:psi"].to_list()))
        self.att_pitch = np.array(data["ATTITUDE:theta"].to_list())

        tnorm_mask = np.array(self.gvf_tnorm) > 0

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

        # Parameters of "Wait and Draw level set"
        self.wait_t = self.tdata[tnorm_mask][0]

        phi_ls = self.gvf_phi[tnorm_mask][0]
        delta = np.sqrt(phi_ls + 1)

        gvf_traj_ls = GvfEllipse(
            self.gvf_traj.XYoff,
            self.gvf_traj.alpha,
            delta * self.gvf_traj.a,
            delta * self.gvf_traj.b,
        )

        self.wait_its = int(wait_period * self.fps)
        self.anim_frames += self.wait_its

        # -----------------------------------------------------------------------------
        # # Run a simulation and collect the predicted data
        # x0 = [
        #     np.array([[self.xdata[tnorm_mask][0], self.ydata[tnorm_mask][0]]]),  # p0
        #     np.array([self.gps_speed[tnorm_mask][0]]),  # v0
        #     np.array([self.att_yaw[tnorm_mask][0]]),  # phi0
        # ]

        # sim = simulator(x0=x0, dt=dt, **sim_kw_args)

        # tf = self.tdata[tnorm_mask][-1] - self.tdata[tnorm_mask][0]

        # t_list = np.arange(0, tf, dt)
        # for it in tqdm(range(len(t_list)), desc="Generating predicted data"):
        #     sim.v = np.array([self.gps_speed[tnorm_mask][it]])
        #     sim.int_euler()

        # self.p_pred = np.array(sim.data["p_pred"])[:, 0, :]

        # -----------------------------------------------------------------------------
        # Initialize the plot and axis configuration
        self.fig = plt.figure(dpi=dpi, figsize=figsize)
        grid = plt.GridSpec(3, 5, hspace=0.2, wspace=0.25)
        self.ax = self.fig.add_subplot(grid[:, 0:3])
        self.ax_phi = self.fig.add_subplot(grid[0, 3:5])
        self.ax_roll = self.fig.add_subplot(grid[1, 3:5])

        if xlims is not None:
            self.ax.set_xlim(xlims)
        if ylims is not None:
            self.ax.set_ylim(ylims)

        xmin, xmax = np.min([-0.2, np.min(self.tdata) - 0.2]), np.max(
            [0.2, np.max(self.tdata) + 0.2]
        )
        ymin_phi, ymax_phi = np.min([-0.1, np.min(self.gvf_phi) - 0.1]), np.max(
            [0.1, np.max(self.gvf_phi) + 0.1]
        )
        ymin_roll, ymax_roll = np.min([-0.2, np.min(self.att_roll) - 0.2]), np.max(
            [0.2, np.max(self.att_roll) + 0.2]
        )

        self.ax_phi.set_xlim([xmin, xmax])
        self.ax_roll.set_xlim([xmin, xmax])
        self.ax_phi.set_ylim([ymin_phi, ymax_phi])
        self.ax_roll.set_ylim([ymin_roll, ymax_roll])

        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$  [L]")
        self.ax.set_aspect("equal")
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(50 / 4))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
        self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(50 / 4))
        self.ax.grid(True)

        self.ax_phi.set_ylabel(r"$\phi$")
        self.ax_phi.yaxis.tick_right()
        self.ax_phi.xaxis.set_major_locator(ticker.MultipleLocator(10))
        self.ax_phi.xaxis.set_minor_locator(ticker.MultipleLocator(10 / 4))
        self.ax_roll.yaxis.set_major_locator(ticker.MultipleLocator(ytick_sep_phi))
        self.ax_roll.yaxis.set_minor_locator(ticker.MultipleLocator(ytick_sep_phi / 4))
        self.ax_phi.grid(True)

        self.ax_roll.set_xlabel(r"$t$ [T]")
        self.ax_roll.set_ylabel(r"$\theta$ [rad]")
        self.ax_roll.yaxis.tick_right()
        self.ax_roll.xaxis.set_major_locator(ticker.MultipleLocator(10))
        self.ax_roll.xaxis.set_minor_locator(ticker.MultipleLocator(10 / 4))
        self.ax_roll.yaxis.set_major_locator(ticker.MultipleLocator(ytick_sep_roll))
        self.ax_roll.yaxis.set_minor_locator(ticker.MultipleLocator(ytick_sep_roll / 4))
        self.ax_roll.grid(True)

        self.kw_patch = {"color": kw_color, "size": kw_sz, "lw": kw_lw}
        self.tail_frames = 2000
        # -----------------------------------------------------------------------------
        # Draw misc
        # self.ax.text()

        # -----------------------------------------------------------------------------
        # Draw the trajectory the level set
        self.gvf_traj.draw(self.fig, self.ax, lw=1.4, draw_field=False)

        # Draw level set
        self.traj_levelset = gvf_traj_ls.draw(
            self.fig, self.ax, lw=1, draw_field=False, color="r"
        )
        self.traj_levelset.set_alpha(0)

        # Draw predicted trayectory (perfect IK-GVF following)
        # (self.traj_pred,) = self.ax.plot(
        #     self.p_pred[:, 0],
        #     self.p_pred[:, 1],
        #     c="orange",
        #     ls="--",
        #     lw=0.8,
        # )

        # Initialize agent's icon
        icon_init = fixedwing_patch(
            [self.xdata[0], self.ydata[0]], self.att_yaw[0], **self.kw_patch
        )
        self.agent_icon = fixedwing_patch(
            [self.xdata[0], self.ydata[0]], self.att_yaw[0], **self.kw_patch
        )

        icon_init.set_alpha(kw_alphainit)
        self.agent_icon.set_zorder(10)

        self.ax.add_patch(icon_init)
        self.ax.add_patch(self.agent_icon)

        # Initialize agent's tail
        (self.agent_line,) = self.ax.plot(
            self.xdata[0], self.ydata[0], c=kw_color, ls="-", lw=tail_lw
        )

        # -----------------------------------------------------------------------------
        # Draw PHI data line
        self.ax_phi.axhline(0, color="k", ls="-", lw=1)
        (self.line_phi,) = self.ax_phi.plot(0, self.gvf_phi[0], lw=1.4, zorder=8)

        # Draw PHI predicted line
        phi0 = self.gvf_phi[self.gvf_tnorm > 0][0]
        phi_time = self.tdata[self.gvf_tnorm > 0]

        t = phi_time - phi_time[0]
        self.phi_pred = phi0 * np.exp(-self.ke * t) + A * np.sin(omega * (t + t0_sin))

        (self.line_phi_pred,) = self.ax_phi.plot(
            phi_time,
            self.phi_pred,
            c="orange",
            ls="--",
            lw=1,
        )

        # Draw PHI level set lines

        self.line_ls = self.ax_phi.axhline(
            phi_ls,
            c="red",
            ls="--",
            lw=1,
        )

        self.line_ls_t = self.ax_phi.axvline(
            phi_time[0],
            c="k",
            ls="--",
            lw=0.8,
        )

        # Draw ROLL data line
        self.ax_roll.axhline(0, color="k", ls="-", lw=1)
        (self.line_roll,) = self.ax_roll.plot(0, self.att_roll[0], lw=1.4, zorder=8)

        self.line_roll_t = self.ax_roll.axvline(
            phi_time[0],
            c="k",
            ls="--",
            lw=0.8,
        )

        # Adjust alpha for the wait sequence
        self.line_ls.set_alpha(0)
        self.line_ls_t.set_alpha(0)
        if self.wait_its > 0:
            # self.traj_pred.set_alpha(0)
            self.line_phi_pred.set_alpha(0)
            self.line_roll_t.set_alpha(0)

    # ---------------------------------------------------------------------------------

    def animate(self, iframe):

        # Wait sequence
        if self.wait_its > 0:
            if self.wait_state == 0 and self.tdata[self.last_i] >= self.wait_t:
                self.wait_state = 1
            elif self.wait_state == 1 and self.waited_its >= self.wait_its:
                self.wait_state = 2

            if self.wait_state == 1:
                self.waited_its += 1

                # Update alpha of the level set
                alpha = np.min([1, 3 * self.waited_its / self.wait_its])
                self.traj_levelset.set_alpha(alpha)
                self.line_ls.set_alpha(alpha)
                self.line_ls_t.set_alpha(alpha)
                self.line_roll_t.set_alpha(alpha)

                alpha = np.max([0, 3 * self.waited_its / self.wait_its - 1.1])
                alpha = np.min([1, alpha])
                # self.traj_pred.set_alpha(alpha)
                self.line_phi_pred.set_alpha(alpha)

                i = self.last_i
            else:
                i = iframe - self.waited_its
        else:
            i = iframe

        # Update the icon
        self.agent_icon.remove()
        self.agent_icon = fixedwing_patch(
            [self.xdata[i], self.ydata[i]], self.att_yaw[i], **self.kw_patch
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

        # Update phi and roll data
        self.line_phi.set_data(self.tdata[0:i], self.gvf_phi[0:i])
        self.line_roll.set_data(self.tdata[0:i], self.att_roll[0:i])

        # Save last iteration
        self.last_i = i

    # ---------------------------------------------------------------------------------

    def gen_animation(self):
        """
        Generate the animation object.
        """

        self.wait_state = 0
        self.waited_its = 0
        self.last_i = 0

        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=tqdm(
                range(self.anim_frames),
                initial=1,
                position=0,
                desc="Generating animation frames",
            ),
            interval=1 / self.fps * 1000,
        )
        anim.embed_limit = 40

        # Close plots and return the animation class to be compiled
        plt.close()
        return anim


#######################################################################################

