"""Target tracking environment placeholder."""
from __future__ import annotations

from smart_captain.simulation.compat import BASE_CONFIG, BaseEnvironment
from smart_captain.skills.base import SkillAdapter
from smart_captain.skills.target_tracking.config import MPC_CONFIG, TARGET_TRACKING_SPEC
from scipy.interpolate import interp1d, CubicSpline
import numpy as np

class TargetTrackingEnv(BaseEnvironment,SkillAdapter):
    """Placeholder target tracking environment."""
    spec = TARGET_TRACKING_SPEC

    def __init__(self, env_config: dict = BASE_CONFIG, auv = None, train_mode = True, mpc_config: dict = MPC_CONFIG):
        """
        路径跟踪任务分为两种模式：
        1、测试模式，它将通过傅里叶级数生成一个随机的周期性路径，用于测试这套方法的准确度
        2、执行模式，直接输入steps作为参考路径
        """
        SkillAdapter.__init__(self, runtime_env=auv)
        BaseEnvironment.__init__(self, env_config, auv, train_mode)

        self.N = mpc_config["horizon"]
        self.dt = mpc_config["dot_t"]
        self.v_cruise = mpc_config["v_cruise"]

        self._ref_arc = 0.0          
        self._ref_step = 0           
        self._lookahead = 0.2      

    def _reset_reference_progress(self):
        """重置参考进度，在环境 reset 时调用"""
        self._ref_arc = 0.0
        self._ref_step = 0
    
    def _generate_random_fourier_path(self, seed=42):
        """
        生成一条从 start_pos 到 goal_pos 的随机傅里叶曲线，用于测试路径跟踪
        返回可调用的曲线函数 p(s) 和其导数 dpds(s)，以及曲线总长度 L。
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 随机选择谐波数量 5~10
        n_harm = np.random.randint(5, 11)
        # 频率范围 [1, 5]
        freqs = np.random.uniform(1.0, 5.0, size=n_harm)
        # 振幅范围 [0, 2] 米
        amps_x = np.random.uniform(0, 10, size=n_harm)
        amps_y = np.random.uniform(0, 10, size=n_harm)
        amps_z = np.random.uniform(0, 10, size=n_harm)
        # 正弦/余弦相位
        sin_phase_x = np.random.uniform(0, 2*np.pi, size=n_harm)
        cos_phase_x = np.random.uniform(0, 2*np.pi, size=n_harm)
        sin_phase_y = np.random.uniform(0, 2*np.pi, size=n_harm)
        cos_phase_y = np.random.uniform(0, 2*np.pi, size=n_harm)
        sin_phase_z = np.random.uniform(0, 2*np.pi, size=n_harm)
        cos_phase_z = np.random.uniform(0, 2*np.pi, size=n_harm)
        
        # 包络函数 w(s)=s*(1-s)
        envelope = lambda s: s * (1 - s)
        
        # 定义扰动函数
        def disturbance(s):
            s_arr = np.atleast_1d(s)
            dx = np.zeros_like(s_arr)
            dy = np.zeros_like(s_arr)
            dz = np.zeros_like(s_arr)
            for i in range(n_harm):
                dx += amps_x[i] * (np.sin(2*np.pi*freqs[i]*s_arr + sin_phase_x[i]) +
                                   np.cos(2*np.pi*freqs[i]*s_arr + cos_phase_x[i]))
                dy += amps_y[i] * (np.sin(2*np.pi*freqs[i]*s_arr + sin_phase_y[i]) +
                                   np.cos(2*np.pi*freqs[i]*s_arr + cos_phase_y[i]))
                dz += amps_z[i] * (np.sin(2*np.pi*freqs[i]*s_arr + sin_phase_z[i]) +
                                   np.cos(2*np.pi*freqs[i]*s_arr + cos_phase_z[i]))
            return np.stack([dx, dy, dz], axis=-1) * envelope(s_arr)[:, np.newaxis]
        
        # 直线基线方向
        base_dir = self.goal_pos - self.start_pos
        # 完整曲线: p(s) = start_pos + base_dir * s + disturbance(s)
        def p(s):
            s_arr = np.atleast_1d(s)
            base = self.start_pos + base_dir * s_arr[:, np.newaxis]
            return base + disturbance(s_arr)
        
        # 导数 dp/ds 的解析形式（需对扰动求导）
        def dpds(s):
            s_arr = np.atleast_1d(s)
            # 基线导数
            base_deriv = base_dir  # 常数向量
            # 扰动导数：包络导数 w'(s)=1-2s，乘以扰动函数 + 扰动本身的导数 * w(s)
            w_prime = 1 - 2*s_arr
            # 计算扰动本身的导数（对s）
            d_dist = np.zeros((len(s_arr), 3))
            for i in range(n_harm):
                arg = 2*np.pi*freqs[i]*s_arr
                d_sin = 2*np.pi*freqs[i] * np.cos(arg + sin_phase_x[i])
                d_cos = -2*np.pi*freqs[i] * np.sin(arg + cos_phase_x[i])
                d_dist[:, 0] += amps_x[i] * (d_sin + d_cos)
                
                d_sin = 2*np.pi*freqs[i] * np.cos(arg + sin_phase_y[i])
                d_cos = -2*np.pi*freqs[i] * np.sin(arg + cos_phase_y[i])
                d_dist[:, 1] += amps_y[i] * (d_sin + d_cos)
                
                d_sin = 2*np.pi*freqs[i] * np.cos(arg + sin_phase_z[i])
                d_cos = -2*np.pi*freqs[i] * np.sin(arg + cos_phase_z[i])
                d_dist[:, 2] += amps_z[i] * (d_sin + d_cos)
            
            # 包络函数值
            w_val = envelope(s_arr)[:, np.newaxis]
            # 扰动值本身
            dist_val = disturbance(s_arr)
            
            deriv = base_deriv[np.newaxis, :] + d_dist * w_val + dist_val * w_prime[:, np.newaxis]
            return deriv
        
        # 计算曲线总长度 L = ∫_0^1 |dp/ds| ds
        s_samples = np.linspace(0, 1, 500)
        deriv_samples = dpds(s_samples)
        ds = s_samples[1] - s_samples[0]
        speed_s = np.linalg.norm(deriv_samples, axis=1)
        L = np.trapezoid(speed_s, s_samples)
        self.T_total = L / self.v_cruise
        return p, dpds, L
    
    def _generate_straight_path(self):

        base_dir = self.goal_pos - self.start_pos
        L = np.linalg.norm(base_dir)   # 直线长度
    
        def p(s):
            s_arr = np.atleast_1d(s)
            # 直线插值
            pos = self.start_pos + base_dir * s_arr[:, np.newaxis]
            return pos
    
        def dpds(s):
            s_arr = np.atleast_1d(s)
            # 导数为常数向量，形状 (len(s), 3)
            deriv = np.tile(base_dir, (len(s_arr), 1))
            return deriv
        self.T_total = L / self.v_cruise
        return p, dpds, L

    def _path_from_steps(self, steps, current_pos):
        # 确保起点与当前AUV位置一致
        steps = steps.copy()
        steps[0] = current_pos
    
        # 参数化：按累计弦长归一化 s ∈ [0,1]
        diffs = np.diff(steps, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate(([0], np.cumsum(seg_lengths)))
        total_len = cum_len[-1]
        s_param = cum_len / total_len   # 离散点对应的s值
    
        # 对每个维度分别进行三次样条插值
        cs_x = CubicSpline(s_param, steps[:, 0], bc_type='clamped')
        cs_y = CubicSpline(s_param, steps[:, 1], bc_type='clamped')
        cs_z = CubicSpline(s_param, steps[:, 2], bc_type='clamped')
    
        def p(s):
            s_arr = np.atleast_1d(s)
            x = cs_x(s_arr)
            y = cs_y(s_arr)
            z = cs_z(s_arr)
            return np.stack([x, y, z], axis=-1)
    
        def dpds(s):
            s_arr = np.atleast_1d(s)
            dx = cs_x.derivative()(s_arr)
            dy = cs_y.derivative()(s_arr)
            dz = cs_z.derivative()(s_arr)
            return np.stack([dx, dy, dz], axis=-1)
    
        # 总长度直接用 total_len（注意：样条曲线长度会略有不同，但可作为参考）
        L = total_len
        self.T_total = L / self.v_cruise
        return p, dpds, L
    
    def _generate_circular_path(self):
        C = self.goal_pos                 # 圆心
        P = self.start_pos                # 圆上当前点
        v = P - C                         # 从圆心指向起点的向量
        R = np.linalg.norm(v)            # 圆半径
        v_unit = v / R

        # 判断是否平行于 XOY 平面
        is_parallel_to_xoy = np.abs(v[2]) < 1e-6

        if is_parallel_to_xoy:
            # 水平圆：法向量沿 Z 轴
            n = np.array([0.0, 0.0, 1.0])
        else:
            # 一般情况：需满足法向量 n ⟂ v 且 n·(v × k) = 0
            k = np.array([0.0, 0.0, 1.0])
            v_z = v[2]
            n_raw = v_z * v - R**2 * k     # n = (v·k)v - (v·v)k
            n_norm = np.linalg.norm(n_raw)
            if n_norm < 1e-6:
                # 竖直方向退化：v ≈ (0,0,±R)，此时任何包含竖直线的平面均可。
                # 我们取法向量为 (1,0,0) 对应的平面（即 x-z 平面）
                n = np.array([1.0, 0.0, 0.0])
            else:
                n = n_raw / n_norm

        # 局部坐标系：u 为径向单位向量（圆心 -> 起点），w 为切线方向，形成右手系
        u = v_unit
        w = np.cross(n, u)                # 保证 w 在圆平面内且与 u 正交

        # 圆周长
        L = 2.0 * np.pi * R

        def p(s):
            s_arr = np.atleast_1d(s)
            angle = 2.0 * np.pi * s_arr    # s∈[0,1] 对应角度 0 → 2π
            # 圆心 + 极坐标旋转
            pos = C + R * (np.cos(angle)[:, np.newaxis] * u + np.sin(angle)[:, np.newaxis] * w)
            return pos

        def dpds(s):
            s_arr = np.atleast_1d(s)
            angle = 2.0 * np.pi * s_arr
            # dp/ds = d(angle)/ds * (-R sin(angle) u + R cos(angle) w) = 2πR * ( -sin u + cos w )
            deriv = 2.0 * np.pi * R * (-np.sin(angle)[:, np.newaxis] * u + np.cos(angle)[:, np.newaxis] * w)
            return deriv

        # 设置任务总时间（若需要被 T_total 等使用）
        self.T_total = L / self.v_cruise
        return p, dpds, L

    def _find_closest_s(self, p, s_samples, current_pos):
        """
        在预采样的s点上计算曲线位置，找到离当前AUV位置最近的s值。
        p: 曲线函数，输入s数组返回位置数组 (N,3)
        s_samples: 预先计算的s采样点（例如500个等间距点）
        current_pos: 当前AUV位置 (3,)
        返回最近的s值 (float)
        """
        # 计算所有采样点对应的位置
        pos_samples = p(s_samples)  # shape (len(s_samples), 3)
        # 计算每个采样点到当前位置的欧氏距离平方
        dist2 = np.sum((pos_samples - current_pos)**2, axis=1)
        idx = np.argmin(dist2)
        return s_samples[idx]

    def _generate_ref_from_curve_pos(self, p, dpds, L, current_pos, lookahead=0.1):
        """
        基于弧长投影的参考轨迹生成，通过步进方式更新 self._ref_step 和 self._ref_arc。
        """
        # ---------- 弧长参数化缓存 ----------
        if not hasattr(self, '_arc_cache'):
            s_samples = np.linspace(0, 1, 500)
            deriv_samples = dpds(s_samples)
            ds = s_samples[1] - s_samples[0]
            speed_s = np.linalg.norm(deriv_samples, axis=1)
            arc_length = np.cumsum(speed_s) * ds
            arc_length = np.concatenate(([0], arc_length[:-1]))
            total_arc = arc_length[-1]
            s_from_arc = interp1d(arc_length, s_samples, kind='linear', 
                                fill_value=(0,1), bounds_error=False)
            self._arc_cache = {
                's_samples': s_samples,
                'arc_length': arc_length,
                'total_arc': total_arc,
                's_from_arc': s_from_arc
            }
        cache = self._arc_cache
        s_samples = cache['s_samples']
        arc_length = cache['arc_length']
        total_arc = cache['total_arc']
        s_from_arc = cache['s_from_arc']

        # ---------- 1. 计算投影弧长 ----------
        s_closest = self._find_closest_s(p, s_samples, current_pos)
        arc_proj = np.interp(s_closest, s_samples, arc_length)

        # ---------- 2. 步进式更新参考进度 ----------
        step_arc = self.v_cruise * self.dt   # 每步前进弧长

        # 若 AUV 超前，逐步增加 ref_step 直到参考弧长 >= arc_proj - lookahead
        while arc_proj > self._ref_arc + self._lookahead:
            self._ref_arc = min(self._ref_arc + step_arc, total_arc)
            self._ref_step += 1
            if self._ref_arc >= total_arc:
                break

        # 正常情况：AUV 在参考点后方一个前视距离内，则前进一步
        if arc_proj > self._ref_arc - self._lookahead:
            self._ref_arc = min(self._ref_arc + step_arc, total_arc)
            self._ref_step += 1
        # 否则（AUV落后太多），保持当前 ref_step 不变

        # ---------- 3. 生成未来 N+1 步参考状态 ----------
        arc_start = min(self._ref_arc + self._lookahead, total_arc)
        s_start = float(s_from_arc(arc_start))

        ref = np.zeros((self.N+1, 12))
        for k in range(self.N+1):
            arc = arc_start + self.v_cruise * (k * self.dt)
            if arc >= total_arc:
                s = 1.0
            else:
                s = float(s_from_arc(arc))

            pos = p(s).flatten()
            deriv = dpds(np.array([s])).flatten()
            norm_deriv = np.linalg.norm(deriv)
            tangent = deriv / norm_deriv if norm_deriv > 1e-6 else np.array([1,0,0])

            psi = np.arctan2(tangent[1], tangent[0])
            theta = np.arctan2(tangent[2], np.sqrt(tangent[0]**2 + tangent[1]**2))
            phi = 0.0

            v_inertial = self.v_cruise * tangent
            c_phi, s_phi = np.cos(phi), np.sin(phi)
            c_theta, s_theta = np.cos(theta), np.sin(theta)
            c_psi, s_psi = np.cos(psi), np.sin(psi)
            R_bi = np.array([
                [c_theta*c_psi, c_theta*s_psi, -s_theta],
                [s_phi*s_theta*c_psi - c_phi*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, s_phi*c_theta],
                [c_phi*s_theta*c_psi + s_phi*s_psi, c_phi*s_theta*s_psi - s_phi*c_psi, c_phi*c_theta]
            ])
            v_body = R_bi.T @ v_inertial
            omega_body = np.zeros(3)

            ref[k, 0:3] = pos
            ref[k, 3:6] = [phi, theta, psi]
            ref[k, 6:9] = v_body
            ref[k, 9:12] = omega_body

        return ref
    
    def _generate_ref_from_curve_t(self, p, dpds, L, current_step):
        """给定曲线 p(s), dpds(s) 和总长度 L，生成参考状态序列。"""
    
        s_samples = np.linspace(0, 1, 500)
        deriv_samples = dpds(s_samples)
        ds = s_samples[1] - s_samples[0]
        speed_s = np.linalg.norm(deriv_samples, axis=1)
        arc_length = np.cumsum(speed_s) * ds
        arc_length = np.concatenate(([0], arc_length[:-1]))
        total_arc = arc_length[-1]
        s_from_arc = interp1d(arc_length, s_samples, kind='linear', fill_value=(0,1), bounds_error=False)
    
        ref = np.zeros((self.N+1, 12))
        for k in range(self.N+1):
            t = (current_step + k) * self.dt
            if t >= self.T_total:
                s = 1.0
            else:
                arc = self.v_cruise * t
                s = float(s_from_arc(np.clip(arc, 0, total_arc)))
        
            pos = p(s).flatten()
            deriv = dpds(np.array([s])).flatten()
            norm_deriv = np.linalg.norm(deriv)
            tangent = deriv / norm_deriv if norm_deriv > 1e-6 else np.array([1,0,0])
        
            psi = np.arctan2(tangent[1], tangent[0])
            theta = np.arctan2(tangent[2], np.sqrt(tangent[0]**2 + tangent[1]**2))
            phi = 0.0
        
            v_inertial = self.v_cruise * tangent
            c_phi, s_phi = np.cos(phi), np.sin(phi)
            c_theta, s_theta = np.cos(theta), np.sin(theta)
            c_psi, s_psi = np.cos(psi), np.sin(psi)
            R_bi = np.array([
                [c_theta*c_psi, c_theta*s_psi, -s_theta],
                [s_phi*s_theta*c_psi - c_phi*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, s_phi*c_theta],
                [c_phi*s_theta*c_psi + s_phi*s_psi, c_phi*s_theta*s_psi - s_phi*c_psi, c_phi*c_theta]
            ])
            v_body = R_bi.T @ v_inertial
            omega_body = np.zeros(3)
        
            ref[k, 0:3] = pos
            ref[k, 3:6] = [phi, theta, psi]
            ref[k, 6:9] = v_body
            ref[k, 9:12] = omega_body
    
        return ref

    def generate_reference_trajectory(self, current_pos, goal_pos, path_steps=None, use_straight_line=False):
        """
        根据当前模式调用对应的参考生成函数。
        统一使用基于位置 + 进度记忆的方法（_generate_ref_from_curve_pos）。
        """
        if path_steps is not None:
            p, dpds, L = self._path_from_steps(path_steps, current_pos)
            return self._generate_ref_from_curve_pos(p, dpds, L, current_pos)
        else:
            cache_key = tuple(goal_pos)
            if not hasattr(self, '_path_cache'):
                self._path_cache = {}
            if cache_key not in self._path_cache:
                p, dpds, L = self._generate_random_fourier_path()
                self._path_cache[cache_key] = (p, dpds, L)
            else:
                p, dpds, L = self._path_cache[cache_key]
            return self._generate_ref_from_curve_pos(p, dpds, L, current_pos)

    #MPC修改
    def build_tracking_observation(self):
        if not hasattr(self, "goal_pos") or self.goal_pos is None:
            self.goal_pos = self.goal_location

        if not hasattr(self, "start_pos") or self.start_pos is None:
            self.start_pos = self.auv_position

        if not hasattr(self, "_path_cache"):
            self._path_cache = {}

        ref = self.generate_reference_trajectory(self.auv_position, self.goal_pos)

        states = np.zeros(12)
        states[0:3] = self.auv_position
        states[3:6] = self.auv_attitude
        states[6:9] = self.auv_relative_velocity
        states[9:12] = self.auv_angular_velocity

        self.ref = ref
        return np.concatenate([states, ref.flatten()])
    #MPC修改结束

    def reset(self, seed=None, return_info=True, options=None):
        obs, info_dict = super().reset(seed, return_info, options)

        self.goal_pos = self.goal_location
        self.start_pos = self.auv_position
        self.path_steps = None

        self._path_cache = {}
        # 重置参考进度
        self._reset_reference_progress()

        p, dpds, L = self._generate_random_fourier_path(seed=42)
        cache_key = tuple(self.goal_location)
        self._path_cache[cache_key] = (p, dpds, L)
        ref = self._generate_ref_from_curve_pos(p, dpds, L, self.auv_position)
        states = np.zeros(12)
        states[0:3] = self.auv_position
        states[3:6] = self.auv_attitude
        states[6:9] = self.auv_relative_velocity
        states[9:12] = self.auv_angular_velocity
        obs = np.concatenate([states, ref.flatten()])
        self.ref = ref
        return obs, info_dict

    def step(self, action):
        _, reward, done, _, info = super().step(action)

        obs = self.build_tracking_observation()
        
        return obs, reward, done, False, self.info

    def get_done(self):
        # 当参考进度到达终点且AUV也接近终点时判定结束
        if self._ref_arc >= self._arc_cache['total_arc'] - 0.1:
            dist_to_goal = np.linalg.norm(self.auv_position - self.goal_pos)
            if dist_to_goal < 1.0 and np.linalg.norm(self.auv_relative_velocity) < 0.2:
                return True
        return False
    
    def print_path(self):
        print("\nBeginning to print ref:")
        for i in range(self.N + 1):
            goal_state = self.ref[i]
            print("\ngoal state:")
            print(f"  pos: {goal_state[0:3]}")
            print(f"  att (rad): {goal_state[3:6]}")
            print(f"  vel: {goal_state[6:9]}")
            print(f"  angvel (rad/s): {goal_state[9:12]}")
    
    def transfer(self, last_task, return_info=True, options=None):
        obs, info_dict = super().transfer(last_task, return_info, options)
        
        self.goal_pos = self.goal_location
        self.start_pos = self.auv_position
        self.path_steps = None

        self._path_cache = {}
        # 重置参考进度
        self._reset_reference_progress()

        p, dpds, L = self._generate_random_fourier_path(seed=42)
        cache_key = tuple(self.goal_location)
        self._path_cache[cache_key] = (p, dpds, L)
        ref = self._generate_ref_from_curve_pos(p, dpds, L, self.auv_position)
        states = np.zeros(12)
        states[0:3] = self.auv_position
        states[3:6] = self.auv_attitude
        states[6:9] = self.auv_relative_velocity
        states[9:12] = self.auv_angular_velocity
        obs = np.concatenate([states, ref.flatten()])
        self.ref = ref
        return obs, info_dict
    