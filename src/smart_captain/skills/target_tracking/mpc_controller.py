import numpy as np
from smart_captain.skills.target_tracking.config import MPC_CONFIG
import casadi as ca

class AUV_MPC:
    def __init__(self, config=MPC_CONFIG):
        self.config = config
        self.dt = self.config["dot_t"]
        self.N = self.config["horizon"]
        self.nx = self.config["obs_space"]
        self.nu = self.config["command_space"]
        
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
    
        # action_to_command 矩阵 (8x4)
        self.M_action = np.array([
            [0,0,1,0],
            [0,0,1,0],
            [0,0,1,0],
            [0,0,1,0],
            [1,-1,0,1],
            [1,1,0,-1],
            [1,-1,0,-1],
            [1,1,0,1]
        ], dtype=np.float64)
        # 有效控制分配矩阵 (6x4)
        self.B_eff = self.B @ self.M_action   # 形状 (6,4)
        #self.B_eff = np.array([
            #[2.8,  0.0, 0.0, 0.0],   # surge -> +Fx
            #[0.0, -2.8, 0.0, 0.0],   # sway  -> -Fy, matches HoloOcean response
            #[0.0,  0.0, 4.0, 0.0],   # heave -> +Fz
            #[0.0,  0.0, 0.0, 0.0],   # roll torque ignored
            #[0.0,  0.0, 0.0, 0.0],   # pitch torque ignored
            #[0.0,  0.0, 0.0, 0.12],  # yaw -> +Tz
        #], dtype=np.float64)


        self.Q = self.config["Q"]
        self.R = self.config["R"]
        self.Rd = self.config["Rd"]

        self.u_min = self.config["u_min"]
        self.u_max = self.config["u_max"]
        self.x_min, self.x_max = self.config["x_min"], self.config["x_max"]

        self.u_opt = None

        self._build_symbolic_model()
        self._build_optimization_problem()

    def _state_error(self, x, ref):
        pos_err = x[0:3] - ref[0:3]
        angle_err = ca.vertcat(
            ca.atan2(ca.sin(x[3] - ref[3]), ca.cos(x[3] - ref[3])),
            ca.atan2(ca.sin(x[4] - ref[4]), ca.cos(x[4] - ref[4])),
            ca.atan2(ca.sin(x[5] - ref[5]), ca.cos(x[5] - ref[5])),
        )
        vel_err = x[6:9] - ref[6:9]
        omega_err = x[9:12] - ref[9:12]
        return ca.vertcat(pos_err, angle_err, vel_err, omega_err)

    def _build_symbolic_model(self):
        x_sym = ca.MX.sym('x', self.nx)
        u_sym = ca.MX.sym('u', self.nu)

        pos = x_sym[0:3]
        euler = x_sym[3:6]
        vel = x_sym[6:9]
        omega = x_sym[9:12]
        phi, theta, psi = euler[0], euler[1], euler[2]

        F_tau = ca.mtimes(ca.DM(self.B_eff), u_sym)
        F_ctrl = F_tau[:3]
        tau_ctrl = F_tau[3:]

        c_phi = ca.cos(phi)
        s_phi = ca.sin(phi)
        c_theta = ca.cos(theta)
        s_theta = ca.sin(theta)
        c_psi = ca.cos(psi)
        s_psi = ca.sin(psi)

        R = ca.vertcat(
            ca.horzcat(c_theta*c_psi, s_phi*s_theta*c_psi - c_phi*s_psi, c_phi*s_theta*c_psi + s_phi*s_psi),
            ca.horzcat(c_theta*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, c_phi*s_theta*s_psi - s_phi*c_psi),
            ca.horzcat(-s_theta,      s_phi*c_theta,                       c_phi*c_theta)
        )

        T = ca.vertcat(
            ca.horzcat(1, s_phi*ca.tan(theta), c_phi*ca.tan(theta)),
            ca.horzcat(0, c_phi,              -s_phi),
            ca.horzcat(0, s_phi/c_theta,       c_phi/c_theta)
        )

        e3_inertial = ca.vertcat(0, 0, 1)
        e3_body = R.T @ e3_inertial
        F_rest = self.deltaB * e3_body
        tau_rest = ca.cross(self.r_B, self.buoyancy * e3_body) + ca.cross(self.r_G, -self.W * e3_body)

        F_damp = -self.d_lin * self.m * vel
        tau_damp = -self.d_ang * (self.I @ omega)

        dv = (F_ctrl + F_damp + F_rest) / self.m - ca.cross(omega, vel)
        tau_total = tau_ctrl + tau_damp + tau_rest
        gyro = ca.cross(omega, self.I @ omega)
        domega = self.I_inv @ (tau_total - gyro)

        vel_next = vel + self.dt * dv
        omega_next = omega + self.dt * domega
        pos_next = pos + self.dt * (R @ vel)
        euler_next = euler + self.dt * (T @ omega)

        euler_next = ca.fmod(euler_next + np.pi, 2*np.pi) - np.pi

        next_state = ca.vertcat(pos_next, euler_next, vel_next, omega_next)

        self.f_dyn = ca.Function('f_dyn', [x_sym, u_sym], [next_state])

    def _build_optimization_problem(self):
        X = ca.MX.sym('X', self.nx * (self.N + 1))
        U = ca.MX.sym('U', self.nu * self.N)
        x0_param = ca.MX.sym('x0', self.nx)
        ref_param = ca.MX.sym('ref', (self.N + 1) * self.nx)

        cost = 0
        g_eq = []   # 等式约束列表
        g_ineq = [] # 不等式约束列表

        # 初始状态等式约束（向量）
        g_eq.append(X[:self.nx] - x0_param)

        for k in range(self.N):
            x_k = X[k*self.nx : (k+1)*self.nx]
            u_k = U[k*self.nu : (k+1)*self.nu]
            x_k1 = X[(k+1)*self.nx : (k+2)*self.nx]
            ref_k = ref_param[k*self.nx : (k+1)*self.nx]

            # 跟踪代价
            err = self._state_error(x_k, ref_k)
            cost += ca.mtimes([err.T, self.Q, err])
            # 控制代价
            cost += ca.mtimes([u_k.T, self.R, u_k])
            # 控制增量代价
            if k > 0:
                du = u_k - U[(k-1)*self.nu : k*self.nu]
                cost += ca.mtimes([du.T, self.Rd, du])

            # 动力学等式约束
            x_next_pred = self.f_dyn(x_k, u_k)
            g_eq.append(x_k1 - x_next_pred)

            # 不等式约束：控制限幅（要求 >= 0）
            g_ineq.append(u_k - self.u_min)   # u_k >= u_min
            g_ineq.append(self.u_max - u_k)   # u_k <= u_max
            # 状态限幅
            #g_ineq.append(x_k - self.x_min)   # x_k >= x_min
            #g_ineq.append(self.x_max - x_k)   # x_k <= x_max

        # 终端代价
        x_N = X[self.N*self.nx:]
        ref_N = ref_param[self.N*self.nx:]
        err_N = self._state_error(x_N, ref_N)
        cost += ca.mtimes([err_N.T, self.Q, err_N])

        # 将所有约束拼接成一个列向量
        g_all = ca.vertcat(*g_eq, *g_ineq)

        # 获取等式约束的总标量个数（每个 g_eq 元素可能为向量）
        n_eq = ca.vertcat(*g_eq).size1()
        n_ineq = ca.vertcat(*g_ineq).size1()
        total_cons = n_eq + n_ineq

        # 设置约束边界：等式部分上下界均为 0，不等式部分下界 0 上界 +inf
        lbg = np.zeros(total_cons)
        ubg = np.zeros(total_cons)
        lbg[:n_eq] = 0
        ubg[:n_eq] = 0
        lbg[n_eq:] = 0
        ubg[n_eq:] = np.inf

        # 构建 NLP，使用参数 p
        nlp = {
            'x': ca.vertcat(X, U),
            'f': cost,
            'g': g_all,
            'p': ca.vertcat(x0_param, ref_param)
        }
        opts = {'ipopt.max_iter': 200, 'ipopt.print_level': 0, 'ipopt.tol': 1e-4, 'ipopt.acceptable_tol': 1e-6,'ipopt.hessian_approximation': 'limited-memory', 'print_time': 0}
        """
        opts = {
            'ipopt': {
                'max_iter': 200,       # 适当增加最大迭代次数
                'print_level': 5,      # 设置为5，以在控制台输出详细的求解过程
                'tol': 1e-4,           # 容忍度
                'acceptable_tol': 1e-6,
            },
            'print_time': True
        }
        """
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # 保存边界和变量长度信息
        self.lbg = lbg
        self.ubg = ubg
        self.X_len = self.nx * (self.N + 1)
        self.U_len = self.nu * self.N
    
    def predict(self, state):
        """
        state: 一维数组 [current_state (nx), ref_states.flatten()] 长度 nx*(N+2)
        """
        current_state = state[:self.nx]
        ref_flat = state[self.nx:]   # 长度 (N+1)*nx
        ref_states = ref_flat.reshape(self.N+1, self.nx)
        
        # 参数向量
        p_val = np.concatenate([current_state, ref_flat])

        if self.u_opt is not None:
            u_guess_2d = np.roll(self.u_opt, -1, axis=0) 
            u_guess_2d[-1, :] = 0
            u_guess = u_guess_2d.flatten()  
        else:
            u_guess = np.zeros(self.N * self.nu)
        
        # 初始猜测（可选，可以用上一次的解热启动，这里简单用0）
        x_guess = np.zeros((self.N+1, self.nx))
        x_guess[0] = current_state

        if self.u_opt is not None:
            for k in range(self.N):
                x_next = self.f_dyn(x_guess[k], u_guess_2d[k])
                x_guess[k+1] = x_next.full().flatten()
        else:
            for k in range(1, self.N+1):
                # 位置：当前位置 + 参考速度 * 时间
                x_guess[k, 0:3] = current_state[0:3] + ref_states[0, 6:9] * (k * self.dt)
                # 速度：直接使用参考速度（假设很快达到）
                x_guess[k, 6:9] = ref_states[0, 6:9]
                # 姿态：从参考轨迹中获取（或保持当前）
                x_guess[k, 3:6] = ref_states[k, 3:6] if k < len(ref_states) else ref_states[-1, 3:6]
                # 角速度：0
                x_guess[k, 9:12] = 0
        x_guess = x_guess.flatten()

        x0_guess = np.concatenate([x_guess, u_guess])
        
        # 求解
        sol = self.solver(x0=x0_guess, p=p_val, lbg=self.lbg, ubg=self.ubg)
        u_opt = sol['x'].full().flatten()[self.X_len:]

        self.u_opt = u_opt.reshape(-1, self.nu)
        return u_opt[:self.nu]


    def action_test(self, state):
        print("\nPredictor's test is beginning")
        state = state[:self.nx]
        for i in range(self.N):
            predicted_state = self.f_dyn(state, self.u_opt[i])
            print("\nPredicted next state:")
            print(f"  pos: {predicted_state[0:3]}")
            print(f"  att (rad): {predicted_state[3:6]}")
            print(f"  vel: {predicted_state[6:9]}")
            print(f"  angvel (rad/s): {predicted_state[9:12]}")
            state = predicted_state
