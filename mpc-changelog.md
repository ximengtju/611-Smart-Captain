## 新增文件

在 `611-Smart-Captain-new-main/src/smart_captain/skills/target_tracking/` 下新增了 MPC 相关文件：
config.py
env.py
policy.py
evaluator.py
mpc_controller.py
mpc_predictor.py

来源关系：

`HoloOcean-2.0.1/MPC/mpc_controller.py` -> `skills/target_tracking/mpc_controller.py`

`HoloOcean-2.0.1/MPC/mpc_predictor.py` -> `skills/target_tracking/mpc_predictor.py`

`HoloOcean-2.0.1/task/task3.py` -> `skills/target_tracking/env.py`

`HoloOcean-2.0.1/env/env_config.py` -> `/skills/target_tracking/config`



## 修改内容
`611-Smart-Captain-new-main/src/smart_captain/skills/registry.py`
去除了target_tracking的占位，registry技能注册现在位于 `611-Smart-Captain-new-main/src/smart_captain/skills/target_tracking/config.py`行51-65

`611-Smart-Captain-new-main/src/smart_captain/app/bridge.py`
行34 将target_tracking模式设为2
行51-54 target_tracking部分补全

`611-Smart-Captain-new-main/src/smart_captain/app/compat_runtime.py`
行61-90 增加MPC和RL兼容部分，防止试图调用不存在的RL模型
行183-188 增加当运行同时含有MPC和RL的任务时，能够正确的转换obs的维度数
行232-239 274-281 增加切换为MPC模式的判断，防止试图调用不存在的RL模型

`611-Smart-Captain-new-main/src/smart_captain/orchestration/evaluator_registry.py`
行25 增加target_tracking的evaluator_registry


