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

__all__ = ["AnimationXYPhi"]

class AnimationXYPhi:
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
        self.phidata = np.array(data["phi"].tolist())[:,0]

        self.ke = np.array(data["ke"].tolist())[0]
        self.A = np.array(data["gamma_A"].tolist())[0]
        self.omega = np.array(data["gamma_omega"].tolist())[0]

        self.ut_norm = np.array(data["ut_norm"].tolist())[:,0]
        # self.p_pred = np.array(data["p_pred"].tolist())[:,:,:]

        # Parameters of "Wait and Draw level set"
        self.wait_t = self.tdata[self.ut_norm > 0][0]

        phi_ls = self.phidata[self.ut_norm > 0][0]
        delta = np.sqrt(phi_ls + 1)

        gvf_traj_ls = gvf_traj

        # -----------------------------------------------------------------------------
        # Initialize the plot and axis configuration
        self.fig = plt.figure(**kw_fig)
        grid = plt.GridSpec(3, 5, hspace=0.1, wspace=0.25)
        self.ax = self.fig.add_subplot(grid[:, 0:3])
        self.ax_phi = self.fig.add_subplot(grid[0, 3:5])

        # XY plot
        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$  [L]")
        self.ax.set_aspect("equal")
        config_data_axis(self.ax, **kw_ax)

        # Phi plot
        self.ax_phi.set_xlabel(r"$t$ [T]")
        self.ax_phi.set_ylabel(r"$\phi$")

        xmin, xmax = np.min([-0.2, np.min(self.tdata) - 0.2]), np.max(
            [0.2, np.max(self.tdata) + 0.2]
        )
        ymin, ymax = np.min([-1, np.min(self.phidata) - 1]), np.max(
            [1, np.max(self.phidata) + 1]
        )
        self.ax_phi.set_xlim([xmin, xmax])
        self.ax_phi.set_ylim([ymin, ymax])
        config_data_axis(self.ax, x_step=5, y_step=2, y_right=True)

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
        #     self.p_pred[:, 0, 0],
        #     self.p_pred[:, 0, 1],
        #     c="orange",
        #     ls="--",
        #     lw=0.8,
        # )
        # self.traj_pred.set_alpha(0)

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
        # Draw PHI data line
        self.ax_phi.axhline(0, color="k", ls="-", lw=1)
        (self.line_phi,) = self.ax_phi.plot(0, self.phidata[0], lw=1.4, zorder=8)

        # Draw PHI predicted line
        phi0 = self.phidata[self.ut_norm > 0][0]
        phi_time = self.tdata[self.ut_norm > 0]

        t = phi_time - phi_time[0]
        self.phi_pred = phi0 * np.exp(-self.ke * t) + self.A * np.sin(self.omega * t)

        (self.line_phi_pred,) = self.ax_phi.plot(
            phi_time,
            self.phi_pred,
            c="orange",
            ls="--",
            lw=1,
        )

        self.line_phi_pred.set_alpha(0)

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

        self.line_ls.set_alpha(0)
        self.line_ls_t.set_alpha(0)

        # -----------------------------------------------------------------------------

    def animate(self, iframe):

        # Wait sequence
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

            alpha = np.max([0, 3 * self.waited_its / self.wait_its - 1.1])
            alpha = np.min([1, alpha])
            # self.traj_pred.set_alpha(alpha)
            self.line_phi_pred.set_alpha(alpha)

            i = self.last_i
        else:
            i = iframe - self.waited_its

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

        # Update phi data
        self.line_phi.set_data(self.tdata[0:i], self.phidata[0:i])

        # Save last iteration
        self.last_i = i

    def gen_animation(self, fps=None, anim_tf=None, tail_frames=500, wait_period=3):
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

        # Set wait period
        self.wait_its = int(wait_period * self.fps)
        self.anim_frames += self.wait_its

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
