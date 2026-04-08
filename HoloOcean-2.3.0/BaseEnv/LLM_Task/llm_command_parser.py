import json
import numpy as np


SYSTEM_PROMPT = """
你是水下机器人导航任务解析器。
你必须把用户输入转换成严格 JSON。
只能输出 JSON，不要输出解释，不要输出 markdown。

只允许以下 3 种 task_type：

1) 绝对坐标导航
{
  "task_type": "navigate_to_point",
  "target": [x, y, z]
}

2) 相对方向导航（机体系方向 + 总距离）
{
  "task_type": "move_relative_direction",
  "direction": "front",
  "distance": 0.0
}

3) 无法解析
{
  "task_type": "reject",
  "reason": "..."
}

direction 只能取以下值之一：
- front
- back
- left
- right
- left_front
- right_front
- left_back
- right_back
- up
- down
- left_front_down
- right_front_down
- left_back_down
- right_back_down

规则：
- “导航到/前往/去 全局坐标 x y z” => navigate_to_point
- “前进 N 米” => move_relative_direction, direction="front", distance=N
- “后退 N 米” => move_relative_direction, direction="back", distance=N
- “左移 N 米” => move_relative_direction, direction="left", distance=N
- “右移 N 米” => move_relative_direction, direction="right", distance=N
- “下潜 N 米” => move_relative_direction, direction="down", distance=N
- “上浮 N 米” => move_relative_direction, direction="up", distance=N
- “左前方 N 米” => move_relative_direction, direction="left_front", distance=N
- “右前方 N 米” => move_relative_direction, direction="right_front", distance=N
- “左后方 N 米” => move_relative_direction, direction="left_back", distance=N
- “右后方 N 米” => move_relative_direction, direction="right_back", distance=N
- “左前下方 N 米” => move_relative_direction, direction="left_front_down", distance=N
- “右前下方 N 米” => move_relative_direction, direction="right_front_down", distance=N

非常重要：
- “左前方 10 米” 表示沿左前方向总共移动 10 米
- 不是前进 10 米再左移 10 米
- distance 表示最终合成位移的总长度
- 如果句子中包含“看看有没有沉船/搜索一下”等额外语义，但当前系统只有导航能力，优先保留导航部分
- 如果没有明确距离或明确坐标，输出 reject
- 坐标和距离必须是数字
"""


class QwenCommandParser:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def parse(self, text: str, current_pos=None, current_yaw=None) -> dict:
        payload = {
            "instruction": text,
            "current_position_world": (
                current_pos.tolist() if isinstance(current_pos, np.ndarray) else current_pos
            ),
            "current_yaw_rad": current_yaw,
            "note": "当前环境中 z 越小表示越深，例如 -40 比 -5 更深。"
        }

        raw = self.llm_client.chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False)
        )

        task = json.loads(raw)
        self._validate_task(task)
        return task

    def _validate_task(self, task: dict):
        task_type = task.get("task_type")

        if task_type == "navigate_to_point":
            target = task.get("target")
            if not isinstance(target, list) or len(target) != 3:
                raise ValueError(f"LLM 输出的 target 非法: {task}")

        elif task_type == "move_relative_direction":
            direction = task.get("direction")
            distance = task.get("distance")

            valid_directions = {
                "front", "back", "left", "right",
                "left_front", "right_front", "left_back", "right_back",
                "up", "down",
                "left_front_down", "right_front_down",
                "left_back_down", "right_back_down"
            }

            if direction not in valid_directions:
                raise ValueError(f"LLM 输出的 direction 非法: {task}")

            if distance is None:
                raise ValueError(f"LLM 输出缺少 distance: {task}")

            task["distance"] = float(distance)

        elif task_type == "reject":
            if "reason" not in task:
                task["reason"] = "unknown"

        else:
            raise ValueError(f"不支持的 task_type: {task_type}")


def resolve_goal_from_task(task: dict, current_pos: np.ndarray, current_yaw: float) -> np.ndarray:
    """
    把结构化任务转换成世界坐标目标点
    current_pos: 世界坐标 [x, y, z]
    current_yaw: 当前航向（弧度）
    """
    task_type = task["task_type"]

    if task_type == "navigate_to_point":
        return np.array(task["target"], dtype=np.float32)

    if task_type == "move_relative_direction":
        direction = task["direction"]
        distance = float(task["distance"])

        direction_map = {
            "front": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "back": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "left": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "right": np.array([0.0,-1.0, 0.0], dtype=np.float32),

            "left_front": np.array([1.0, 1.0, 0.0], dtype=np.float32),
            "right_front": np.array([1.0,-1.0, 0.0], dtype=np.float32),
            "left_back": np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            "right_back": np.array([-1.0, 1.0, 0.0], dtype=np.float32),

            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "down": np.array([0.0, 0.0, -1.0], dtype=np.float32),

            "left_front_down": np.array([1.0, -1.0, -1.0], dtype=np.float32),
            "right_front_down": np.array([1.0, 1.0, -1.0], dtype=np.float32),
            "left_back_down": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            "right_back_down": np.array([-1.0, 1.0, -1.0], dtype=np.float32),
        }

        v_body = direction_map[direction]
        v_body = v_body / np.linalg.norm(v_body) * distance

        forward = v_body[0]
        right = v_body[1]
        dz_body = v_body[2]

        cy = np.cos(current_yaw)
        sy = np.sin(current_yaw)

        # 机体系 -> 世界系（水平面）
        dx = cy * forward - sy * right
        dy = sy * forward + cy * right

        # 这里保持和你当前环境定义一致：
        # z 越小越深，因此 body 中 down 对应世界坐标 z 减小
        dz = dz_body

        goal = current_pos + np.array([dx, dy, dz], dtype=np.float32)
        return goal

    if task_type == "reject":
        raise ValueError(task.get("reason", "LLM 拒绝解析该命令"))

    raise ValueError(f"未知任务类型: {task_type}")