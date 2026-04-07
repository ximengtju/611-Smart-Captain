import csv
import numpy as np
from stable_baselines3 import SAC


from BaseEnv.Env_Task4.env.openwater_env import BaseEnvironment
from BaseEnv.Env_Task4.env.env_config import BASE_CONFIG


MODEL_PATH = r"F:\OceanGym\HoloOcean2\BaseEnv\Barrier_Task4\logs1\task1-v0_SAC_1\task1-v0_SAC_1500000.zip"   # 改成你的模型路径
SAVE_CSV = r"eval_one_episode.csv"


GOAL = np.array([190.0, -110.0, -275.0], dtype=np.float32)


def main():
    env = BaseEnvironment(BASE_CONFIG)
    env.set_goal(GOAL)

    model = SAC.load(MODEL_PATH, env=env)

    obs, info = env.reset()
    done = False
    truncated = False
    ep_reward = 0.0

    rows = []

    print("===== Start Eval Episode =====")
    print("goal =", GOAL.tolist())

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        ep_reward += reward
        d_min = float(np.min(env.radar_intersec_dist))
        speed = float(np.linalg.norm(env.auv_relative_velocity))
        ang_rate = float(np.linalg.norm(env.auv_angular_velocity))

        row = {
            "t_step": info["t_step"],
            "x": float(info["position"][0]),
            "y": float(info["position"][1]),
            "z": float(info["position"][2]),
            "delta_d": float(info["delta_d"]),
            "delta_psi": float(env.delta_psi),
            "speed": speed,
            "ang_rate": ang_rate,
            "radar_min": d_min,
            "reward": float(reward),
            "cum_reward": float(ep_reward),
            "done": bool(done),
            "goal_reached": bool(info["goal_reached"]),
            "collision": bool(info["collision"]),
            "done_reason": "|".join(info["conditions_true_info"]) if done else "",
            "a_surge": float(action[0]),
            "a_sway": float(action[1]),
            "a_heave": float(action[2]),
            "a_yaw": float(action[3]),
        }
        rows.append(row)

        if info["t_step"] % 50 == 0 or done or truncated:
            print(
                f"step={info['t_step']:4d} "
                f"pos=({row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f}) "
                f"delta_d={row['delta_d']:.3f} "
                f"speed={row['speed']:.3f} "
                f"ang_rate={row['ang_rate']:.3f} "
                f"radar_min={row['radar_min']:.3f} "
                f"reward={row['reward']:.3f}"
            )

    print("\n===== Episode Finished =====")
    print("ep_reward =", ep_reward)
    print("goal_reached =", info["goal_reached"])
    print("collision =", info["collision"])
    print("done_reason =", info["conditions_true_info"])
    print("final_position =", info["position"])
    print("final_delta_d =", info["delta_d"])

    with open(SAVE_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"trajectory saved to: {SAVE_CSV}")


if __name__ == "__main__":
    main()