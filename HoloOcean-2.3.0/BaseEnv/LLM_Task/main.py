import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import time
import numpy as np
import gymnasium as gym
from collections import deque
from BaseEnv.stable_baselines3 import SAC
from Env_Task1.env.env_config import BASE_CONFIG

from llm_client import QwenClient
from llm_command_parser import QwenCommandParser, resolve_goal_from_task


MODEL_PATH = r"F:\OceanGym\HoloOcean2\BaseEnv\Env_Task1\logs1\task1-v0_SAC_1\task1-v0_SAC_1000000.zip"
MAX_DEMO_STEPS = 4000
STEP_SLEEP = 0.01


def get_current_state(base_env):
    current_pos = np.array(base_env.auv_position, dtype=np.float32)
    current_yaw = float(base_env.auv_attitude[2])
    return current_pos, current_yaw


def print_runtime_stats(step_count, inference_times, sim_times, base_env):
    avg_inference = np.mean(inference_times) * 1000
    avg_sim = np.mean(sim_times) * 1000
    total_time = avg_inference + avg_sim
    fps = 1000 / total_time if total_time > 1e-6 else 0.0

    print(
        f"step={step_count:4d} | "
        f"推理:{avg_inference:5.1f}ms | "
        f"仿真:{avg_sim:5.1f}ms | "
        f"总:{total_time:5.1f}ms | "
        f"FPS:{fps:5.1f} | "
        f"pos={np.round(base_env.auv_position, 1)} | "
        f"dist={base_env.delta_d:.2f}"
    )


def run_navigation_episode(env_inst, model, goal):
    base_env = env_inst.unwrapped

    # 设置目标
    base_env.set_goal(goal)

    # 某些环境中 goal 需要 reset 后才生效
    obs, info = env_inst.reset()

    print(f"当前坐标: {np.round(base_env.auv_position, 2)}")
    print(f"目标坐标: {np.round(base_env.goal_location, 2)}")

    done = False
    step_count = 0

    inference_times = deque(maxlen=1000)
    sim_times = deque(maxlen=1000)

    print("\n开始执行...")
    print("-" * 60)

    while not done and step_count < MAX_DEMO_STEPS:
        inference_start = time.time()
        action, _ = model.predict(obs, deterministic=True)
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)

        sim_start = time.time()
        obs, reward, terminated, truncated, info = env_inst.step(action)
        sim_time = time.time() - sim_start
        sim_times.append(sim_time)

        done = terminated or truncated
        step_count += 1

        if step_count % 50 == 0:
            print_runtime_stats(step_count, inference_times, sim_times, base_env)

        time.sleep(STEP_SLEEP)

    print("-" * 60)
    print("\n【执行完成】")

    if len(inference_times) > 0:
        avg_inf = np.mean(inference_times) * 1000
        avg_sim = np.mean(sim_times) * 1000
        total_avg = avg_inf + avg_sim
        fps = 1000 / total_avg if total_avg > 1e-6 else 0.0

        print(f"平均推理时间: {avg_inf:.1f} ms")
        print(f"平均仿真时间: {avg_sim:.1f} ms")
        print(f"平均总时间: {total_avg:.1f} ms")
        print(f"平均FPS: {fps:.1f}")

        if avg_sim > avg_inf * 2:
            print("➤ 瓶颈在【仿真交互】")
        elif avg_inf > avg_sim * 2:
            print("➤ 瓶颈在【算法推理】")
        else:
            print("➤ 算法和仿真相对平衡")

    result = {
        "final_position": np.array(base_env.auv_position, dtype=np.float32),
        "goal_position": np.array(base_env.goal_location, dtype=np.float32),
        "distance_error": float(base_env.delta_d),
        "steps": step_count,
        "done": done,
    }

    print(f"\n最终位置: {np.round(result['final_position'], 2)}")
    print(f"目标位置: {np.round(result['goal_position'], 2)}")
    print(f"最终距离误差: {result['distance_error']:.2f}")

    return result


def main():
    env_config = BASE_CONFIG.copy()
    env_inst = gym.make("task1-v0", env_config=env_config)
    model = SAC.load(MODEL_PATH)

    llm = QwenClient(model="qwen-plus")  # 从 DASHSCOPE_API_KEY 读取 key
    parser = QwenCommandParser(llm)

    print("环境、模型、Qwen 解析器加载完成。")
    print("支持示例：")
    print("  前往左前方 30m 查看有无沉船出现")
    print("  导航到全局坐标 x=-30 y=20 z=-15")
    print("  前进 10 米")
    print("  下潜 3 米")
    print("输入 quit 退出。")

    while True:
        cmd = input("\n请输入指令：").strip()
        if cmd.lower() in ["quit", "exit", "q"]:
            break

        # 先 reset 一次，获取当前状态用于解析相对指令
        obs, info = env_inst.reset()
        base_env = env_inst.unwrapped
        base_env.AUV.move_viewport(
            location = [0,0,-3],
            rotation = [0,0,0],
        )
        base_env.AUV.tick()
        current_pos, current_yaw = get_current_state(base_env)

        print(f"解析前当前位置: {np.round(current_pos, 2)}")
        print(f"解析前当前yaw: {current_yaw:.3f} rad")

        try:
            task = parser.parse(cmd, current_pos=current_pos, current_yaw=current_yaw)
            print(f"LLM解析任务: {task}")

            goal = resolve_goal_from_task(task, current_pos, current_yaw)
            print(f"解析后的目标点: {np.round(goal, 2)}")

        except Exception as e:
            print(f"指令解析失败: {e}")
            continue

        # 记录 reset 前位置，后面检查是否一致
        pos_before_goal_reset = np.array(base_env.auv_position, dtype=np.float32)

        result = run_navigation_episode(env_inst, model, goal)

        pos_after_goal_reset = result["final_position"] * 0 + np.array(base_env.auv_position, dtype=np.float32)

        # 提醒：如果 reset 会随机起点，相对指令会失真
        if np.linalg.norm(pos_before_goal_reset - np.array(base_env.auv_position, dtype=np.float32)) > 1e-3:
            print("\n[警告] 你的环境在 set_goal 后 reset 改变了起点。")
            print("       对“前进10米 / 下潜3米”这类相对指令，目标可能和解析时的起点不一致。")
            print("       后续建议改成：更新目标后不 reset，或提供一个专门刷新 goal 的接口。")

    env_inst.close()


if __name__ == "__main__":
    main()