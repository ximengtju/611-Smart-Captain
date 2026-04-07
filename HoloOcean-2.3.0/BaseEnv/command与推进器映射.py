import numpy as np
import holoocean
import copy

scenario_cfg = {
    "name": "HoveringMapTest",
    "world": "PierHarbor",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 200,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "DynamicsSensor",
                    "socket": "COM",
                    "configuration": {
                        "UseCOM": True,
                        "UseRPY": True
                    }
                },
                {
                    "sensor_type": "VelocitySensor",
                    "socket": "IMUSocket"
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -10],
            "rotation": [0.0, 0.0, 0.0],  # 验证时先设为0度最省事
        }
    ]
}

# =========================
# 2) 你要验证的映射
# action = [surge, sway, heave, yaw]
# thruster order:
# [VFS, VFP, VBP, VBS, AFS, AFP, ABP, ABS]
# =========================
def action_to_command(action, thruster_scale=10.0, lateral_scale=1.0, yaw_scale=1.0):
    surge, sway, heave, yaw = np.asarray(action, dtype=np.float32)

    surge *= thruster_scale
    sway  *= thruster_scale * lateral_scale
    heave *= thruster_scale
    yaw   *= thruster_scale * yaw_scale

    cmd = np.zeros(8, dtype=np.float32)

    # 4个垂直桨
    cmd[0] = heave
    cmd[1] = heave
    cmd[2] = heave
    cmd[3] = heave

    # 4个斜桨
    cmd[4] = surge - sway - yaw   # AFS
    cmd[5] = surge + sway + yaw   # AFP
    cmd[6] = surge - sway + yaw   # ABP
    cmd[7] = surge + sway - yaw   # ABS

    return cmd

def ssa(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def world_to_body(v_world, yaw):
    """
    只用yaw把世界系速度旋回艇体系，足够做 surge/sway 粗验证。
    x_body: 前向
    y_body: 右向/侧向（具体正方向以实验结果为准）
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    R_bw = np.array([
        [ c, s, 0],
        [-s, c, 0],
        [ 0, 0, 1],
    ], dtype=np.float32)
    return R_bw @ v_world

def read_state(sensor_return):
    dyn = np.asarray(sensor_return["DynamicsSensor"], dtype=np.float32)
    vel_world = np.asarray(sensor_return["VelocitySensor"], dtype=np.float32)

    # DynamicsSensor with UseRPY=True:
    # [accel(0:3), vel(3:6), pos(6:9), ang_acc(9:12), ang_vel(12:15), rpy(15:18)]
    pos = dyn[6:9]
    ang_vel = dyn[12:15]
    rpy = dyn[15:18]
    roll, pitch, yaw = rpy
    return pos, vel_world, ang_vel, yaw

def settle(env, n=40):
    state = None
    zero = np.zeros(8, dtype=np.float32)
    for _ in range(n):
        state = env.step(zero)
    return state

def pulse_test(env, action, pulse_steps=30, settle_steps=40):
    # 清零稳定
    state = env.reset()
    state = settle(env, settle_steps)

    pos0, vel0_w, angvel0, yaw0 = read_state(state)

    cmd = action_to_command(action)

    vel_hist_body = []
    yaw_hist = []
    pos_hist = []

    for _ in range(pulse_steps):
        state = env.step(cmd)
        pos, vel_w, ang_vel, yaw = read_state(state)

        vel_b = world_to_body(vel_w, yaw0)   # 用初始yaw转回艇体系
        vel_hist_body.append(vel_b)
        yaw_hist.append(yaw)
        pos_hist.append(pos)

    vel_hist_body = np.asarray(vel_hist_body)
    pos_hist = np.asarray(pos_hist)

    mean_vb = vel_hist_body.mean(axis=0)
    delta_pos = pos_hist[-1] - pos0
    delta_yaw = ssa(yaw_hist[-1] - yaw0)

    return {
        "action": np.asarray(action, dtype=np.float32),
        "command": cmd,
        "mean_body_velocity": mean_vb,
        "delta_position_world": delta_pos,
        "delta_yaw_rad": float(delta_yaw),
    }

if __name__ == "__main__":
    env = holoocean.make(scenario_cfg=copy.deepcopy(scenario_cfg), show_viewport=False)

    tests = {
        "surge+": np.array([+1, 0, 0, 0], dtype=np.float32),
        "sway+":  np.array([0, +1, 0, 0], dtype=np.float32),
        "heave+": np.array([0, 0, +1, 0], dtype=np.float32),
        "yaw+":   np.array([0, 0, 0, +1], dtype=np.float32),
    }

    for name, act in tests.items():
        out = pulse_test(env, act, pulse_steps=30, settle_steps=40)
        print("=" * 60)
        print(name)
        print("action:", out["action"])
        print("thruster command:", np.round(out["command"], 3))
        print("mean body velocity [vx, vy, vz]:", np.round(out["mean_body_velocity"], 4))
        print("delta position world [dx, dy, dz]:", np.round(out["delta_position_world"], 4))
        print("delta yaw [rad]:", round(out["delta_yaw_rad"], 4))
