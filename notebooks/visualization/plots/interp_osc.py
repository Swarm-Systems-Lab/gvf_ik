"""
"""

__all__ = ["PlotInterpOsc"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ssl_simulator.visualization import config_data_axis

#######################################################################################

class PlotInterpOsc:
    def __init__(self, data):
        self.data = data
        self.fig, self.ax = plt.subplots(1,2,figsize=(10,4))

    def config_axes(self, y):
        self.ax[0].set_xlabel(r"$x$ [m]")
        self.ax[0].set_ylabel(r"$y$ [m]")
        self.ax[0].grid(True)

        self.ax[1].set_xlabel(r"$\dot{\bar x}$ [m/s]")
        self.ax[1].set_ylabel(r"$A$ [m]")
        self.ax[1].grid(True)

        # Compute y-axis limits with padding
        y_min, y_max = np.min(y), np.max(y)
        pad_dist = abs(y_max - y_min) * 0.1
        y_lower_pad, y_upper_pad = y_min - pad_dist, y_max + pad_dist

        config_data_axis(self.ax[0], y_right=False)
        config_data_axis(self.ax[1], ylims=[y_lower_pad, y_upper_pad], y_right=False)

        return y_lower_pad, y_upper_pad

    def plot(self):
        """
        Generates a plot based on extracted and processed data.
        NOTE: (SciPy doc) The relationship between cov and parameter error estimates  
              is derived based on a linear approximation to the model function around 
              the optimum. 
              [https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2005WR004804]
        """
        
        # Extract derired data
        x = np.array(self.data["p"].tolist())[:,:,0]
        y = np.array(self.data["p"].tolist())[:,:,1]
        x_dot = np.array(self.data["p_dot"].tolist())[1:, :, 0]
        omega = np.array(self.data["gamma_omega"].tolist())[0,0]
        A = np.array(self.data["gamma_A"].tolist())[0,:]
        speed = np.array(self.data["speed"].tolist())[0,0]
        time_values = np.array(self.data["time"].tolist())[1:]

        N = len(A)

        # Compute the mean velocity ẋ during the first period for each omega
        period = 2 * np.pi / omega
        mask = time_values < period
        x_dot_mean = np.mean(x_dot[mask,:], axis=0)

        # Define the interpolation function
        def ellarc(x, a):
            return a * np.sqrt(speed**2 - x**2) / omega 
        
         # Interpolate A(ẋ) for each omega and estimate parameters
        params, _ = curve_fit(ellarc, x_dot_mean, A)
        print("E =", params[0])
        
        # -----------------------------------------------------------------------------
        # PLOT
        x_lin = np.linspace(np.min(x_dot_mean), np.max(x_dot_mean), 100)

        self.ax[0].plot(x, y, "b", alpha=0.6)
        self.ax[0].scatter(x[0,:], y[0,:], edgecolors="r", marker="s", facecolors="None")
        self.ax[0].scatter(x[-1,:], y[-1,:],edgecolors="r", facecolors="None", zorder=3)

        self.ax[1].scatter(x_dot_mean, A, edgecolors="r", marker="s", facecolors="None")
        self.ax[1].plot(x_lin, ellarc(x_lin, *params), "k", 
                        label=r"$k_A \sqrt{v^2 - \dot{\bar x}^2}/w$")

        # Configure axes for plotting
        y_lower_pad, y_upper_pad = self.config_axes(A)

        # Add vertical reference lines
        self.ax[1].vlines(
            x_dot_mean[: N], y_lower_pad, y_upper_pad,
            color="gray", linestyles="--", linewidth=1, zorder=1, alpha=0.8
        )

        # Add legend
        self.ax[1].legend()

        return self.ax
    
    def save(self, filename, dpi=100):
        plt.savefig(filename, dpi=dpi)

#######################################################################################