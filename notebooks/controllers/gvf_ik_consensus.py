"""
"""

__all__ = ["GvfIKCons"]

import numpy as np
from collections.abc import Iterable

from ssl_simulator import Controller

from ssl_simulator.math import build_B
from .utils import CircularBuffer

A_FIT = 1.35

#######################################################################################

class GvfIKCons(Controller):
    def __init__(self, gvf_traj, edges_list, s, ke, kn, omega, ka, buff_len = 5000):

        # Controller settings
        self.gvf_traj = gvf_traj
        self.edges_list = edges_list 
        self.s = s
        self.ke = ke
        self.kn = kn
        self.ka = ka

        if isinstance(omega, Iterable):
            # Use the first value if omega is an iterable
            self.omega = omega[0]
            print("Warning: 'omega' is an Iterable, but only the first value has been considered.")
        elif isinstance(omega, (int, float)):
            # omega is already a valid value (int or float)
            self.omega = omega
        else:
            raise TypeError("Invalid type for 'omega': Expected int, float, or Iterable.")  

        self.period = 2 * np.pi / self.omega

        # Circular buffer for position data. While not optimal for simulation,
        # this implementation mirrors the structure used in pprz, allowing us 
        # to test the real controller's behavior.
        self.pos_buff = CircularBuffer(int(buff_len), self.period) 

        # TODO: check if the size of the buffer is enough

        # ---------------------------
        # Control-related variables (initialized as None, but may later hold arrays)
        self.phi = None # np.zeros(self.N)
        self.e = None   # np.zeros(self.N)
        self.L_bar = None

        # ---------------------------
        # Controller output variables
        self.control_vars = {
            "u": None,
        }

        # Controller variables to be tracked by logger
        self.tracked_vars = {
            "s": self.s,
            "ke": self.ke,
            "kn": self.kn,
            "ka": self.ka,
            "Z": self.edges_list,
            "gamma_omega": self.omega,
            "gamma_A": None,
            "gamma": None,
            "phi": None,
            "e": None,
            "e_cons": None,

            "alpha": None,
            "omega_d": None,
            "ut_norm": None,
            "un_norm": None,

            "x_traj": None,
            "x_avg_dot_d": None,
        }

        # Controller data
        self.init_data()
    
    def compute_laplacian(self, N, dim):
        B = build_B(self.edges_list, N)
        self.L_bar = np.kron(B @ B.T, np.eye(dim))

    def compute_alpha(self, J1, J2, phi, gamma, gamma_dot, speed):
        """
        """
        J_Jt = (J1*J1 + J2*J2)

        # Compute th feedforward error
        e = phi - gamma
        e_tdot = - gamma_dot

        # Compute the input term of p_dot (normal term)
        u = - self.ke * e

        un_x = J1 / J_Jt * (u - e_tdot)
        un_y = J2 / J_Jt * (u - e_tdot)

        un_norm2 = un_x*un_x + un_y*un_y

        # Return the evaluated condition
        return un_norm2 < speed*speed
    
    def compute_project_onto_line(self, A, B, P):
        # Compute vector AB and AP
        AB = B - A
        AP = P - A

        # Compute squared norm of AB
        AB_norm_sq = np.dot(AB, AB)

        # Compute projection scalar t
        t = np.dot(AB, AP) / AB_norm_sq
        t_bar = t * np.sqrt(AB_norm_sq)  # Scaled projection value

        # Compute the projected point
        P_proj = A + t * AB

        return t_bar, P_proj
    
    def compute_control(self, time, state):
        """
        """
        p = state["p"]
        speed = state["speed"]
        theta = state["theta"]

        N = p.shape[0]

        # GVF trajectory data
        phi_l, J_l, H_l = np.zeros(N), np.zeros((N,2)), np.zeros((N,2,2))
        for i in range(N):
            phi_l[i] = self.gvf_traj[i].phi(p[i,:])
            J_l[i] = self.gvf_traj[i].grad_phi(p[i,:])
            H_l[i] = self.gvf_traj[i].hess_phi(p[i,:])
        
        # Compute the maximum amplitude and the minimum desired velocity
        self.A_max = speed / self.omega
        self.x_dot_min = np.sqrt(A_FIT**2 - 1)/ A_FIT * speed

        if self.L_bar is None:
            self.compute_laplacian(N, 1)

        # Compute the projection of (x,y) onto a line
        x = np.zeros(N)
        for i in range(N):
            pt_A = self.gvf_traj[i].A
            pt_B = self.gvf_traj[i].B
            x[i], _ = self.compute_project_onto_line(pt_A, pt_B, p[i,:])
            
        # Insert new values into the buffer
        self.pos_buff.enqueue(time, x)
        self.tracked_vars["x_traj"] = x

        # Initialize the controller output and the variables to be tracked
        self.tracked_vars["e_cons"] = np.zeros((N))
        self.tracked_vars["x_avg_dot_d"] = np.zeros((N))
        self.tracked_vars["gamma_A"] = np.zeros((N))

        self.control_vars["u"] = np.zeros((N))
        self.tracked_vars["e"] = np.zeros((N))
        self.tracked_vars["alpha"] = np.zeros((N))
        self.tracked_vars["phi"] = np.zeros((N))
        self.tracked_vars["gamma"] = np.zeros((N))
        self.tracked_vars["omega_d"] = np.zeros((N))
        self.tracked_vars["ut_norm"] = np.zeros((N))
        self.tracked_vars["un_norm"] = np.zeros((N))

        # CONSENSUS -------------------------------------------------------------------
        A_ctrl = np.zeros(N) 
        if self.pos_buff.max_period_flag:
            buff_p = np.array(self.pos_buff.get_valid_items())

            p_mean = np.mean(buff_p, 0)
            p_mean_bar = p_mean.flatten()
            
            xij_avg_sum = (self.L_bar @ p_mean_bar).reshape(p_mean.shape)
            xi_avg_dot = speed - self.ka*xij_avg_sum/2 # TODO: divide xij_avg_sum bu the number of neighbors
            xi_avg_dot = np.clip(xi_avg_dot, self.x_dot_min, speed)

            # Calculate the requiered amplitude A(x_avg_dot)
            A_ctrl = A_FIT * np.sqrt(speed**2 - xi_avg_dot**2) / self.omega
            A_ctrl = np.clip(A_ctrl, 0, self.A_max*0.97)

            # -----------------------
            self.tracked_vars["e_cons"] = xij_avg_sum
            self.tracked_vars["x_avg_dot_d"] = xi_avg_dot
            self.tracked_vars["gamma_A"] = A_ctrl
            # -----------------------
            
        if (A_ctrl * self.omega > speed).any():
            raise ValueError("Constraint violated: A_ctrl * omega must be ≤ speed!")

        # GVF_IK ----------------------------------------------------------------------
    
        # TODO: there may be something to fix here...
        
        for i in range(N):
            # -------------------
            # GVF trajectory data
            phi = phi_l[i] # Phi value
            J = J_l[i] # Phi gradient (2,)
            H = H_l[i] # Phi hessian  (2,2)

            speed_i = speed[i]
            theta_i = theta[i]
            A_fd = A_ctrl[i]
            omega_fd = self.omega
            
            s = self.s
            ke = self.ke
            kn = self.kn
            # -------------------

            J1 = J[0]
            J2 = J[1]
            
            H11 = H[0,0]
            H12 = H[0,1]
            H21 = H[1,0]
            H22 = H[1,1]
            
            J_Jt = (J1*J1 + J2*J2)

            # 2. Compute th feedforward error
            gamma = A_fd * np.sin(omega_fd * time)
            gamma_dot = omega_fd * A_fd * np.cos(omega_fd * time)

            cond_flag = self.compute_alpha(J1, J2, phi, gamma, gamma_dot, speed_i)

            if cond_flag:
                e = phi - gamma
                e_tdot = - gamma_dot
                e_tddot = (omega_fd * omega_fd) * gamma
            else:
                e = phi
                e_tdot = 0
                e_tddot = 0
            
            # 3. Compute the input term of p_dot (normal term)
            u = - ke * e

            un_x = J1 / J_Jt * (u - e_tdot)
            un_y = J2 / J_Jt * (u - e_tdot)

            un_norm2 = un_x*un_x + un_y*un_y
            un_norm = np.sqrt(un_norm2)
            un_norm3 = un_norm2 * un_norm

            # 4. Compute alpha and the tangent term of p_dot
            ut_x = s * J2
            ut_y = -s * J1

            ut_norm = np.sqrt(ut_x*ut_x + ut_y*ut_y)
            ut_norm3 = ut_norm * ut_norm * ut_norm
            
            ut_hat_x = ut_x / ut_norm
            ut_hat_y = ut_y / ut_norm
            
            # 5. Compute alpha and p_dot
            if cond_flag:
                alpha = np.sqrt(speed_i*speed_i - un_norm2)

                pd_dot_x = alpha * ut_hat_x + un_x
                pd_dot_y = alpha * ut_hat_y + un_y
            else:
                alpha = 0
                
                pd_dot_x = speed_i * un_x / un_norm
                pd_dot_y = speed_i * un_y / un_norm

            ut_dot_x = s * (H12 * pd_dot_x + H22 * pd_dot_y)
            ut_dot_y = - s * (H11 * pd_dot_x + H21 * pd_dot_y) 

            # 6. Compute un_dot
            u_dot = - ke * (J1*pd_dot_x + J2*pd_dot_y + e_tdot)

            un_dot_A_x = (pd_dot_x*H11 + pd_dot_y*H21)
            un_dot_A_y = (pd_dot_x*H12 + pd_dot_y*H22)

            B_term = 2 * ((H11*pd_dot_x + H21*pd_dot_y)*J1 + (H21*pd_dot_x + H22*pd_dot_y)*J2) / J_Jt
            un_dot_B_x = - J1 * B_term
            un_dot_B_y = - J2 * B_term

            C_term = (u_dot - e_tddot) / J_Jt
            un_dot_C_x = J1 * C_term
            un_dot_C_y = J2 * C_term

            un_dot_x = (un_dot_A_x + un_dot_B_x) * (u - e_tdot) / J_Jt + un_dot_C_x
            un_dot_y = (un_dot_A_y + un_dot_B_y) * (u - e_tdot) / J_Jt + un_dot_C_y

            # 7. Compute omega_d and omega
            if cond_flag:
                alpha_dot = - (un_x*un_dot_x + un_y*un_dot_y) / (alpha)
                Bpd_ddot_x = alpha * (ut_dot_x / ut_norm + (ut_x*ut_x*ut_dot_x - ut_x*ut_y*ut_dot_y) / ut_norm3)
                Bpd_ddot_y = alpha * (ut_dot_y / ut_norm + (ut_x*ut_y*ut_dot_x - ut_y*ut_y*ut_dot_y) / ut_norm3)

                pd_ddot_x = alpha_dot * ut_hat_x + Bpd_ddot_x + un_dot_x
                pd_ddot_y = alpha_dot * ut_hat_y + Bpd_ddot_y + un_dot_y
            else:
                pd_ddot_x = speed_i * (un_dot_x / un_norm + (un_x*un_x*un_dot_x - un_x*un_y*un_dot_y) / un_norm3)
                pd_ddot_y = speed_i * (un_dot_y / un_norm + (un_x*un_y*un_dot_x - un_y*un_y*un_dot_y) / un_norm3)
            
            omega_d = - (- pd_dot_x*pd_ddot_y + pd_dot_y*pd_ddot_x) / (speed_i*speed_i)

            r_x = speed_i * np.cos(theta_i)
            r_y = speed_i * np.sin(theta_i)

            omega = omega_d - kn * (pd_dot_x*r_y - pd_dot_y*r_x) / (speed_i*speed_i)

            # -----------------------
            self.control_vars["u"][i] = omega
            self.tracked_vars["e"][i] = e
            self.tracked_vars["alpha"][i] = cond_flag
            self.tracked_vars["phi"][i] = phi
            self.tracked_vars["gamma"][i] = gamma
            self.tracked_vars["omega_d"][i] = omega_d
            self.tracked_vars["ut_norm"][i] = ut_norm
            self.tracked_vars["un_norm"][i] = un_norm
            # -----------------------

        return self.control_vars
    
    #######################################################################################