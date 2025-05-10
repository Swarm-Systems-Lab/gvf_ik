"""
"""

__all__ = ["GvfIK"]

import numpy as np

from ssl_simulator import Controller

#######################################################################################

class GvfIK(Controller):
    def __init__(self, gvf_traj, s, ke, kn, A, omega):

        # Controller settings
        self.gvf_traj = gvf_traj
        self.s = s
        self.ke = ke
        self.kn = kn

        self.A = A
        self.omega = omega

        # Controller variables
        self.phi = None #np.zeros(self.N)
        self.e = None #np.zeros(self.N)

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
            "gamma_A": self.A,
            "gamma_omega": self.omega,
            "gamma": None,
            "gamma_dot": None,
            "phi": None,
            "e": None,
            "omega_d": None
        }

        self.tracked_settings = {
            "gvf_traj": gvf_traj,
        }

        # Controller data
        self.init_data()
    
    def check_alpha(self, J1, J2, phi, gamma, gamma_dot, speed):
        """
        """
        J_Jt = (J1*J1 + J2*J2)

        # Compute th feedforward error
        e = phi + gamma
        e_tdot = gamma_dot

        # Compute the input term of p_dot (normal term)
        u = - self.ke * e

        un_x = J1 / J_Jt * (u - e_tdot)
        un_y = J2 / J_Jt * (u - e_tdot)

        un_norm2 = un_x*un_x + un_y*un_y

        # Return the evaluated condition
        return un_norm2 < speed*speed
    
    def compute_control(self, time, state):
        """
        """
        p = state["p"]
        speed = state["speed"]
        theta = state["theta"]
        N = p.shape[0]

        if (self.A * self.omega > speed).any():
            raise ValueError("Constraint violated: A * omega must be ≤ speed!")

        self.tracked_vars["gamma"] = np.zeros((N))
        self.tracked_vars["gamma_dot"] = np.zeros((N))
        self.tracked_vars["omega_d"] = np.zeros((N))
        self.control_vars["u"] = np.zeros((N))            
        for i in range(N):
            # -------------------
            # GVF trajectory data
            phi = self.gvf_traj.phi(p[i,:]) # Phi value
            J = self.gvf_traj.grad_phi(p[i,:])   # Phi gradient (2,)
            H = self.gvf_traj.hess_phi(p[i,:])   # Phi hessian  (2,2)

            speed_i = speed[i]
            theta_i = theta[i]
            A_fd = self.A[i]
            omega_fd = self.omega[i]
            
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

            # 2. Compute the feedforward error
            gamma = A_fd * np.sin(omega_fd * time)
            gamma_dot = omega_fd * A_fd * np.cos(omega_fd * time)

            cond_flag = self.check_alpha(J1, J2, phi, gamma, gamma_dot, speed_i)

            if cond_flag:
                e = phi + gamma
                e_tdot = gamma_dot
                e_tddot = - (omega_fd * omega_fd) * gamma
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

            # 6. Compute ut_dot
            ut_dot_x = s * (H12 * pd_dot_x + H22 * pd_dot_y)
            ut_dot_y = - s * (H11 * pd_dot_x + H21 * pd_dot_y) 

            # 7. Compute un_dot
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

            # 8. Compute omega_d and omega
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
            self.tracked_vars["gamma"][i] = gamma
            self.tracked_vars["gamma_dot"][i] = gamma_dot
            self.tracked_vars["omega_d"][i] = omega_d
            self.control_vars["u"][i] = omega
            # -----------------------

        return self.control_vars
    
    #######################################################################################