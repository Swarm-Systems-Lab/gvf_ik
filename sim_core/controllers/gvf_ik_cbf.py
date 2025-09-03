"""
"""

__all__ = ["GvfIK_CBF"]

import numpy as np

from ssl_simulator import Controller

#######################################################################################

class GvfIK_CBF(Controller):
    def __init__(self, gvf_traj, s, ke, kn, obstacles = [], gamma=0.5, col_rad=20):

        # Controller settings
        self.gvf_traj = gvf_traj
        self.s = s
        self.ke = ke
        self.kn = kn

        self.obstacles = obstacles

        self.gamma = gamma
        self.col_rad = col_rad

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
            "gamma": self.gamma,
            "col_rad": self.col_rad,
            #
            "phi": None,
            "e": None,
            "omega_d": None,
            #
            "omega_ref": None,
            "lgh": None,
            "lgh_all": None,
        }

        self.tracked_settings = {
            "gvf_traj": gvf_traj,
        }

        # Controller data
        self.init_data()
    
    def check_alpha(self, J1, J2, phi, speed):
        """
        """
        J_Jt = (J1*J1 + J2*J2)

        # Compute th feedforward error
        e = phi
        e_tdot = 0

        # Compute the input term of p_dot (normal term)
        u = - self.ke * e

        un_x = J1 / J_Jt * (u - e_tdot)
        un_y = J2 / J_Jt * (u - e_tdot)

        un_norm2 = un_x*un_x + un_y*un_y

        # Return the evaluated condition
        return un_norm2 < speed*speed
    
    def compute_cbf(self, state):
        omega_ref = self.control_vars["u"]
        gamma = self.gamma
        col_rad = self.col_rad

        # Extract state variables
        P = state["p"]
        V = state["speed"][:,None] * np.array([np.cos(state["theta"]), np.sin(state["theta"])]).T
        N = P.shape[0]
        
        # Initialize controller variables
        omega_safe = np.zeros(omega_ref.shape)
        lgh = np.zeros([N, N])
        lgh_all = np.zeros([N, N])
        
        # Compute the CBF for each agent
        for i in range(N):
            if i in self.obstacles:
                continue
            v = state["speed"][i,None]
            phi = state["theta"][i]
            vrel_dot_1 = v * np.array([np.sin(phi), -np.cos(phi)])

            psi_lgh_k = []
            for k in [k for k in range(N) if k!=i]:
                # p_rel
                prel = P[k,:] - P[i,:]
                prel_sqr = np.dot(prel, prel)
                prel_norm = np.sqrt(prel_sqr)

                # v_rel
                vrel = V[k,:] - V[i,:]
                vrel_norm = np.sqrt(np.dot(vrel, vrel))

                # vk = state["speed"][k]
                # phik = state["theta"][k]
                vk = 0
                phik = 0

                # If they are not in collision...
                if prel_norm > col_rad and v!=0: 
                    cos_alfa = np.sqrt(prel_norm**2 - col_rad**2)/prel_norm

                    # \dot v_rel (The -w is SO IMPORTANT)
                    vrel_dot_2 = vk * np.array([-np.sin(phik), np.cos(phik)])
                    vrel_dot_ref = vrel_dot_2 * (self.control_vars["u"][k]) + vrel_dot_1 * (omega_ref[i])

                    # Derivative of terms involving A
                    rho_rho_dot = 0

                    # h(x,t)
                    dot_rel = np.dot(prel, vrel)
                    h = dot_rel + prel_norm * vrel_norm * cos_alfa

                    # h_dot_ref(x,t) = h_dot(x, u_ref(x,t))
                    h_dot_ref = vrel_norm**2 + np.dot(prel, vrel_dot_ref) + \
                                    np.dot(vrel, vrel_dot_ref)*(cos_alfa*prel_norm)/vrel_norm + \
                                    vrel_norm * (dot_rel - rho_rho_dot)/(cos_alfa*prel_norm)

                    # psi(x,t)
                    psi = h_dot_ref + gamma * h ** 3


                    # Lgh = grad(h(x,t)) * g(x) = dh/dvrel_dot * vrel_dot
                    Lgh = np.dot(prel + vrel * (cos_alfa*prel_norm)/vrel_norm, vrel_dot_1)  

                    lgh_all[k,i] = Lgh

                    if psi < 0:
                        lgh[k,i] = Lgh
            
                        if self.s ==  1 and Lgh <= 0:
                            pass
                        elif self.s == -1 and Lgh >= 0:
                            pass
                        else:
                            # Explicit solution of the QP problem
                            delta = 0.1
                            if abs(Lgh) > delta:
                                psi_lgh_k.append(- psi / Lgh)

                if len(psi_lgh_k) != 0:
                    omega_min = np.min([np.min(psi_lgh_k), 0])
                    omega_max = np.max([np.max(psi_lgh_k), 0])

                    if abs(omega_min) < abs(omega_max):
                        omega_safe[i] = omega_max
                    else:
                        omega_safe[i] = omega_min

                    self.control_vars["u"][i] += omega_safe[i]

        self.tracked_vars["omega_ref"] = omega_ref
        self.tracked_vars["lgh"] = lgh
        self.tracked_vars["lgh_all"] = lgh_all
        
    def compute_control(self, time, state):
        """
        """
        p = state["p"]
        speed = state["speed"]
        theta = state["theta"]
        N = p.shape[0]

        self.tracked_vars["omega_d"] = np.zeros((N))
        self.control_vars["u"] = np.zeros((N))            
        for i in range(N):
            if i in self.obstacles:
                continue
            # -------------------
            # GVF trajectory data
            phi = self.gvf_traj.phi(p[i,:]) # Phi value
            J = self.gvf_traj.grad_phi(p[i,:])   # Phi gradient (2,)
            H = self.gvf_traj.hess_phi(p[i,:])   # Phi hessian  (2,2)

            speed_i = speed[i]
            theta_i = theta[i]
            
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
            cond_flag = self.check_alpha(J1, J2, phi, speed_i)

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
            self.tracked_vars["omega_d"][i] = omega_d
            self.control_vars["u"][i] = omega
            # -----------------------

        self.compute_cbf(state)
        return self.control_vars
    
    #######################################################################################