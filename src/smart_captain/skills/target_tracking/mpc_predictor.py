import numpy as np
from smart_captain.skills.target_tracking.config import MPC_CONFIG

class HoveringAUVModel:
    def __init__(self, config=MPC_CONFIG):
        self.config = config
        self.dt = self.config["dot_t"]
        
        # Physical parameters
        self.m = self.config["m"]          # kg
        self.g = self.config["g"]
        self.rho = self.config["rho"]
        self.V = self.config["V"]
        self.W = self.m * self.g          # 304.3 N
        self.buoyancy = self.rho * self.V * self.g  # 348.7 N
        self.deltaB = self.buoyancy - self.W      # 44.4 N
        
        # Centers (body frame, meters)
        self.r_G = self.config["r_G"]
        self.r_B = self.config["r_B"]
        
        # Damping
        self.d_lin = self.config["d_lin"]   # 1/s
        self.d_ang = self.config["d_ang"]   # 1/s
        
        # Inertia tensor (estimated)
        self.I = self.config["I"]
        self.I_inv = np.linalg.inv(self.I)
        
        # Thruster positions (relative to CG, meters)
        self.thruster_pos = self.config["thruster_pos"]
        
        # Force directions (body frame, unit thrust)
        self.directions = np.zeros((8, 3))
        for i in range(8):
            if i < 4:
                self.directions[i] = [0, 0, 1]
            else:
                if i % 2 == 0:
                    self.directions[i] = [1/np.sqrt(2), 1/np.sqrt(2), 0]
                else:
                    self.directions[i] = [1/np.sqrt(2), -1/np.sqrt(2), 0]
        
        # Control allocation matrix B (6x8)
        self.B = np.zeros((6, 8))
        for i in range(8):
            f = self.directions[i]
            r = self.thruster_pos[i]
            tau = np.cross(r, f)
            self.B[:3, i] = f
            self.B[3:, i] = tau
    
    def rotation_matrix(self, phi, theta, psi):
        """用于计算AUV的旋转矩阵"""
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        
        R = np.array([
            [c_theta*c_psi, s_phi*s_theta*c_psi - c_phi*s_psi, c_phi*s_theta*c_psi + s_phi*s_psi],
            [c_theta*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, c_phi*s_theta*s_psi - s_phi*c_psi],
            [-s_theta,      s_phi*c_theta,                       c_phi*c_theta]
        ])
        return R
    
    def euler_rates_matrix(self, phi, theta, psi):
        """角速度到欧拉角速率的转换矩阵"""
        c_theta = np.cos(theta)
        # Avoid division by zero
        if abs(c_theta) < 1e-6:
            c_theta = 1e-6 * np.sign(c_theta)
        T = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi),              -np.sin(phi)],
            [0, np.sin(phi)/c_theta,       np.cos(phi)/c_theta]
        ])
        return T
    
    def restore_forces(self, R):
        """恢复力与恢复力矩计算"""
        e3_inertial = np.array([0, 0, 1])
        e3_body = R.T @ e3_inertial   # same as R^T * [0,0,1]
        
        F_rest = self.deltaB * e3_body
        
        # Torque: from buoyancy and gravity acting at different points
        tau_rest = np.cross(self.r_B, self.buoyancy * e3_body) + np.cross(self.r_G, -self.W * e3_body)
        return F_rest, tau_rest
    
    def predict(self, state, command):
        x, y, z, phi, theta, psi, u, v, w, p, q, r = state
        pos = np.array([x, y, z])
        euler = np.array([phi, theta, psi])
        vel = np.array([u, v, w])
        omega = np.array([p, q, r])

        F_ctrl_tau_ctrl = self.B @ command
        F_ctrl = F_ctrl_tau_ctrl[:3] # 控制力计算
        tau_ctrl = F_ctrl_tau_ctrl[3:] # 控制力力矩计算

        R = self.rotation_matrix(phi, theta, psi)
        T = self.euler_rates_matrix(phi, theta, psi)
        F_rest, tau_rest = self.restore_forces(R)

        F_damp = -self.d_lin * self.m * vel # 阻尼力计算
        tau_damp = -self.d_ang * (self.I @ omega) # 阻尼力矩计算

        dv = (F_ctrl + F_damp + F_rest) / self.m - np.cross(omega, vel) 

        tau_total = tau_ctrl + tau_damp + tau_rest
        gyro = np.cross(omega, self.I @ omega)
        domega = self.I_inv @ (tau_total - gyro)

        vel_next = vel + self.dt * dv
        omega_next = omega + self.dt * domega
        pos_next = pos + self.dt * (R @ vel)
        euler_next = euler + self.dt * (T @ omega)

        euler_next = (euler_next + np.pi) % (2*np.pi) - np.pi

        next_state = np.concatenate([pos_next, euler_next, vel_next, omega_next])
        return next_state