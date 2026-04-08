import holoocean, cv2
import sys
from pathlib import Path
import copy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Env_Task3.env.env_config import BASE_CONFIG
scenario = copy.deepcopy(BASE_CONFIG["auv_config"])


# 初始化环境
env = holoocean.make(scenario_cfg=scenario, start_world=True)
env.act('auv0', [10, 10, 10, 10, 0, 0, 0, 0])

# 创建一个 OpenCV 窗口
cv2.namedWindow("Left Camera Output", cv2.WINDOW_NORMAL)
#cv2.namedWindow("Right Camera Output", cv2.WINDOW_NORMAL)

try:
    for _ in range(10000):
        state = env.tick()

        # 显示左相机图像
        if "LeftCamera" in state:
            img_left = state["LeftCamera"][:, :, :3]   # 取前三通道（RGB）
            cv2.imshow("Left Camera Output", img_left)

        # # 显示右相机图像
        # if "RightCamera" in state:
        #     img_right = state["RightCamera"][:, :, :3]   # 取前三通道（RGB）
        #     cv2.imshow("Right Camera Output", img_right)

        # 检测按键，如果按下 ESC 或 Q 键退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
finally:
    cv2.destroyAllWindows()  # 关闭窗口
    env.close()  # 关闭环境