"""
"""

__all__ = ["PlotInterpE"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ssl_simulator.visualization import config_data_axis

#######################################################################################

class PlotInterpE:
    def __init__(self, data, N):
        self.data = data
        self.fig, self.ax = plt.subplots()
        self.N = N

    def config_axes(self, x_step=1, y_step=10):
        """
        Configures the plot axes with appropriate labels, limits, and ticks.
        """
        self.ax.set_xlabel(r"$\omega$ [rad/s]")
        self.ax.set_ylabel(r"$E$")

        # Get current tick values
        x_ticks = self.ax.get_xticks()
        y_ticks = self.ax.get_yticks()

        # Compute tick range
        x_ticks_range = np.ptp(x_ticks)  # Peak-to-peak (max - min)
        y_ticks_range = np.ptp(y_ticks)

        # Adjust axis configuration based on tick range
        if x_ticks_range > 3 * x_step and y_ticks_range > 3 * y_step:
            config_data_axis(self.ax, x_step=x_step, y_step=y_step, y_right=False)
        else:
            print("Warning: The x-y step values may be too large.")
            config_data_axis(self.ax, y_right=False)
        
        #return y_lower_pad, y_upper_pad

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
        
        return rf"$\omega =$ {label}"
    
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

        # Define the interpolation function for A(ẋ)
        def ellarc(x, E):
            return np.sqrt(speed**2 - x**2) * E
            
        # Interpolate A(ẋ) for each omega and estimate parameters
        E_list, sigma_list = [], []
        for i in range(n_omega):
            params, pcov = curve_fit(
                ellarc,
                x_dot_mean[i * self.N : (i + 1) * self.N],
                A[i * self.N : (i + 1) * self.N]
            )
            sigma_approx = np.sqrt(np.diag(pcov))
            E_list.append(params[0])
            sigma_list.append(sigma_approx[0])
        
        # Define the interpolation function for E(\omega)
        def x_inv(x, a):
            return a/x
        
        params, pcov = curve_fit(x_inv, omega_list, E_list)

        # -----------------------------------------------------------------------------
        # PLOT
    
        # Scatter plot of data points
        self.ax.scatter(
            omega_list, E_list, edgecolors="r", marker="o",
            facecolors="None", s=14, zorder=3
        )

        # Plot interpolation curves
        x_lin = np.linspace(np.min(omega_list), np.max(omega_list), 100)
        self.ax.plot(x_lin, x_inv(x_lin, *params), "k")

        # Configure axes for plotting
        self.config_axes()

        # Add text labels
        self.ax.text(omega_list[-1]+0.1, E_list[-1]-0.4, rf"$a = {params[0]:.2f}$")

        # Add legend
        # self.ax.plot([], [], "k", label=r"$\dot{\bar x}(A)$")
        self.ax.plot([], [], "k", label=r"$k_A/\omega$")
        self.ax.legend()
        # -----------------------------------------------------------------------------

        return self.ax
    
    def save(self, filename, dpi=100):
        self.fig.savefig(filename, dpi=dpi)

#######################################################################################