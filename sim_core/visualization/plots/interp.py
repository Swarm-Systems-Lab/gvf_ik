"""
"""

__all__ = ["PlotInterp"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ssl_simulator.visualization import config_data_axis

#######################################################################################

class PlotInterp:
    def __init__(self, data, N, **kw_args):
        self.data = data
        self.fig, self.ax = plt.subplots(**kw_args)
        self.N = N

    def config_axes(self, y, x_step=1, y_step=10):
        """
        Configures the plot axes with appropriate labels, limits, and ticks.
        """
        self.ax.set_xlabel(r"$\dot{\bar x}_i$ [m/s]")
        self.ax.set_ylabel(r"$A_i$ [m]")
        
        # Compute y-axis limits with padding
        y_min, y_max = np.min(y), np.max(y)
        pad_dist = 0.1 * abs(y_max - y_min)
        y_lower_pad, y_upper_pad = y_min - pad_dist, y_max + pad_dist

        # Get current tick values
        x_ticks = self.ax.get_xticks()
        y_ticks = self.ax.get_yticks()

        # Compute tick range
        x_ticks_range = np.ptp(x_ticks)  # Peak-to-peak (max - min)
        y_ticks_range = np.ptp(y_ticks)

        # Adjust axis configuration based on tick range
        if x_ticks_range > 3 * x_step and y_ticks_range > 3 * y_step:
            config_data_axis(self.ax, ylims=[y_lower_pad, y_upper_pad],
                            x_step=x_step, y_step=y_step, y_right=False)
        else:
            print("Warning: The x-y step values may be too large.")
            config_data_axis(self.ax, ylims=[y_lower_pad, y_upper_pad], y_right=False)
        
        return y_lower_pad, y_upper_pad

    def get_omega_label(self, omega):
        """
        Returns a formatted label string for the given omega value.
        """
        if omega < np.pi:
            denominator = np.pi / omega
            if denominator.is_integer():
                label = rf"$\pi/{int(denominator)}$"
            else:
                label = rf"${omega}\pi$"
        elif omega > np.pi:
            numerator = omega / np.pi
            if numerator.is_integer():
                label = rf"${int(numerator)}\pi$"
            else:
                label = rf"${omega}\pi$"
        else:
            label = r"$\pi$"
        
        return rf"$w_\gamma =$ {label}"

    def plot(self):
        """
        Generates a plot based on extracted and processed data.
        NOTE: (SciPy doc) The relationship between cov and parameter error estimates  
              is derived based on a linear approximation to the model function around 
              the optimum. 
              [https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2005WR004804]
        """
        # Extract requiered data
        x_dot = np.array(self.data["p_dot"].tolist())[1:, :, 0]
        omega_data = np.array(self.data["gamma_omega"].tolist())[0, :]
        A = np.array(self.data["gamma_A"].tolist())[0, :]
        speed = np.array(self.data["speed"].tolist())[0, 0]
        time_values = np.array(self.data["time"].tolist())[1:]


        # Determine the number of omega clusters and extract unique omega values
        n_omega = len(omega_data) // self.N
        omega_list = [omega_data[i*self.N] for i in range(n_omega)]

        # Compute the mean velocity ẋ during the first period for each omega
        x_dot_mean = np.zeros_like(omega_data)
        for i, omega in enumerate(omega_list):
            period = 2 * np.pi / omega
            mask = time_values < period
            x_dot_select = x_dot[mask, i * self.N : (i + 1) * self.N]
            x_dot_mean[i * self.N : (i + 1) * self.N] = np.mean(x_dot_select, axis=0)

        # Define the interpolation function
        def ellarc(x, a, omega):
            return a * np.sqrt(speed**2 - x**2) / omega

        # Interpolate A(ẋ) for each omega and estimate parameters
        params_list, sigma_list = [], []
        for i in range(n_omega):
            ellarc_fit = lambda x, a: a * np.sqrt(speed**2 - x**2) / omega_list[i]
            params, pcov = curve_fit(
                ellarc_fit,
                x_dot_mean[i * self.N : (i + 1) * self.N],
                A[i * self.N : (i + 1) * self.N]
            )
            sigma_approx = np.sqrt(np.diag(pcov))
            params_list.append(params[0])
            sigma_list.append(sigma_approx[0])
        
        # Print computed parameters
        for omega, param, sigma in zip(omega_list, params_list, sigma_list):
            print(f"omega = {omega:.2f} --> a = {param:.2f}, sigma = {sigma:.4f}")

        # -----------------------------------------------------------------------------
        # PLOT
    
        # Scatter plot of data points
        self.ax.scatter(
            x_dot_mean, A, edgecolors="r", marker="o",
            facecolors="None", s=14, zorder=3
        )

        # Plot interpolation curves
        x_lin = np.linspace(np.min(x_dot_mean), np.max(x_dot_mean), 100)
        for omega, param in zip(omega_list, params_list):
            self.ax.plot(x_lin, ellarc(x_lin, param, omega), "k")

        # Configure axes for plotting
        y_lower_pad, y_upper_pad = self.config_axes(A)

        # Add vertical reference lines
        self.ax.vlines(
            x_dot_mean[: self.N], y_lower_pad, y_upper_pad,
            color="gray", linestyles="--", linewidth=1, zorder=1
        )

        # Add text labels
        for i in range(n_omega):
            label = self.get_omega_label(omega_list[i])
            if omega_list[i] > np.pi:
                self.ax.text(x_dot_mean[(i+1)*self.N - 1]+0.5, A[(i+1)*self.N - 1]-2.5, label)
            else:
                self.ax.text(x_dot_mean[(i+1)*self.N - 1]+0.5, A[(i+1)*self.N - 1]+1.2, label)

        
        # Add legend
        self.ax.plot([], [], "k", label=r"$k_A \sqrt{v^2 - \dot{\bar x}_i^2}/w_\gamma$")
        self.ax.legend()
        # -----------------------------------------------------------------------------

        return self.ax
    
    def save(self, filename, dpi=100):
        self.fig.savefig(filename, dpi=dpi)

#######################################################################################