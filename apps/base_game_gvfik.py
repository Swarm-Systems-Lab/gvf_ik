import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import tkinter as tk

import matplotlib.pyplot as plt
# # Tell matplotlib to use latex (way more laggy)
# from ssl_simulator.visualization import set_paper_parameters
# set_paper_parameters(fontsize=12)

from ssl_simulator import SimulationEngine, parse_kwargs
from ssl_simulator.visualization import Plotter, config_axis

#######################################################################################

class AppGameGVFIK:
    def __init__(self, fig, ax, 
                 simulator_engine: SimulationEngine, simulator_plotter: Plotter, 
                 **kwargs):
        self.fig, self.ax = fig, ax
        self.simulator_engine = simulator_engine
        self.simulator_plotter = simulator_plotter
        

        # Configure the given axis
        self.ax.set_title("", color="black")
        self.ax.set_xlabel(r"$X$", color="black")
        self.ax.set_ylabel(r"$Y$", color="black")
        config_axis(self.ax, xlims=[-130,130], ylims=[-90,90])

        self.fig.set_facecolor("white")
        self.ax.set_facecolor("white")
        self.ax.tick_params(colors='black')
        for spine in self.ax.spines.values():
            spine.set_color('black')

        self.status_label = tk.Label(
            self.fig.canvas.manager.window, text=f"tf = {simulator_engine.time:.2f}", 
            fg="black", font=("arial", 12))
        self.status_label.pack(anchor="s")

        # Set options
        opts = dict(step_scale = 0.2, step_rot = np.pi/2 * 0.2, fps=20)
        self.opts = parse_kwargs(kwargs, opts)

        # Timers
        self.interval = 1/opts["fps"] # s
        self._sim_timer = self.fig.canvas.new_timer(interval=int(1000*self.interval))
        self._init_sim_timer()

        self._draw_pending = False
        self._draw_delay = 10
        self._draw_timer_running = False
        self._init_draw_timer()

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Track mouse state
        self.is_pressed = [False]

        # Draw the initial plot
        self.simulator_plotter.draw(pts_cond=50)
        plt.show()
        
    # ---------------------------------------------------------------------------------
    # Event Handlers

    def on_press(self, event):
        if event.inaxes == self.ax:
            self.is_pressed[0] = True
            pass

    def on_release(self, event):
        self.is_pressed[0] = False

    def on_motion(self, event):
        if self.is_pressed[0] and event.inaxes == self.ax:
            pass

    def on_scroll(self, event):
        if event.inaxes == self.ax:
            if event.key == 'control':
                if event.button == "up":
                    pass
                elif event.button == "down":
                    pass
            else:
                if event.button == "up":
                    pass
                elif event.button == "down":
                    pass

    def on_key(self, event):
        pass     

    # ---------------------------------------------------------------------------------
    # Main Methods

    def _step_simulation(self):
        self.simulator_engine.run(self.interval, eta=False)
        self.simulator_plotter.update()
        self.status_label.config(text=f"tf = {self.simulator_engine.time:.2f}")

        self._draw_pending = True
        if not self._draw_timer_running:
            self._draw_timer.start()

    # ---------------------------------------------------------------------------------
    #  Timer Methods

    def _init_sim_timer(self):
        self._sim_timer.add_callback(self._step_simulation)
        self._sim_timer.start()

    def _init_draw_timer(self):
        self._draw_timer = self.fig.canvas.new_timer(interval=self._draw_delay)
        self._draw_timer.add_callback(self._draw_if_pending)
    
    def _schedule_canvas_task(self, task):
        self.fig.canvas.manager.window.after(0, task)

    def _draw_if_pending(self):
        if self._draw_pending:
            self.fig.canvas.draw_idle()
            self._draw_pending = False

#######################################################################################