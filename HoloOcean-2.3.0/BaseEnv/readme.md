# BaseEnv
## 项目简介
本项目基于 **HoloOcean** 构建水下机器人强化学习任务环境，包含 4 个任务模块和 1 个大语言模型控制模块：

- `Env_Task1`
- `Env_Task2`
- `Env_Task3`
- `Env_Task4`
- `LLM_Task`

4 个 Task 的整体训练框架基本一致，主要区别在于任务环境、场景文件、奖励设计和附加脚本不同。
---

整体说明：
现在可直接运行的测试代码有LLM_Task的main函数（测试的Env_Task1中的导航任务）以及 Env_Task4中固定起点和固定终点的导航避障任务

LLM_Task的main函数可以调用Env_Task1/2/3/4中的模型，使用时候需接入自己的api接口，导入合适的模型路径（Env_Task1有半成品模型可以直接运行）
其中 Env_Task2中 有举例说明BSTSensor(holoocean版本>=2.2.1)使用以及RGBCamera的使用，暂未引入ImagingSonar
每个Env_Task的场景配置文件（例如Env_Task1中的pierharbor_env.py ）中 都配备了常用传感器以及降维雷达
Env_Task4 涉及到避障任务，已经简单训练固定起点到固定终点的任务模型存放于logs1，后续进行拓展

## 项目目录以及说明

```text
BaseEnv/
├─ Env_Task1/                         # 任务1：强化学习训练与评估
│  ├─ env/                           # 任务1环境核心代码
│  │  ├─ __init__.py                 
│  │  ├─ env_config.py               # 环境配置：场景、传感器、奖励、终止条件等
│  │  ├─ pierharbor_env.py           # PierHarbor 场景环境封装
│  │  ├─ rl_config.py                # 超参数
│  │  ├─ task1.py                    # 任务1环境实现：状态、动作、奖励、终止逻辑
│  │  ├─ trainer.py                  # 通用训练封装：创建环境、模型训练、模型保存
│  │  └─ trainer_ppo.py              # PPO 专用训练封装
│  ├─ logs1/                         # 训练日志、模型权重、TensorBoard 数据
│  ├─ __init__.py                    
│  ├─ evaluate.py                    # 训练后模型评估脚本
│  ├─ train.py                       # 训练入口脚本
│  └─ train_ppo.py                   # PPO 训练入口脚本
│
├─ Env_Task2/                        # 任务2：强化学习训练 + 传感器测试
│  ├─ env/                           # 任务2环境核心代码
│  │  ├─ __init__.py                 
│  │  ├─ env_config.py               # 环境配置文件
│  │  ├─ pierharbor_env.py           # PierHarbor 场景环境封装
│  │  ├─ rl_config.py                # 强化学习超参数配置
│  │  ├─ task2.py                    # 任务2环境实现
│  │  └─ trainer.py                  # 训练封装
│  ├─ logs1/                         # 训练日志与模型保存目录
│  ├─ __init__.py                    
│  ├─ BSTSensor.py                   # BST 传感器测试脚本
│  ├─ RGBCamera.py                   # RGB 相机测试脚本
│  └─ train.py                       # 训练入口
│
├─ Env_Task3/                        # 任务3：强化学习训练 + 大坝场景
│  ├─ env/                           # 任务3环境核心代码
│  │  ├─ __init__.py                
│  │  ├─ Dam_env.py                  # Dam 场景环境封装
│  │  ├─ env_config.py               # 环境配置文件
│  │  ├─ rl_config.py                # 强化学习超参数配置
│  │  ├─ task3.py                    # 任务3环境实现
│  │  └─ trainer.py                  # 训练封装
│  ├─ logs1/                         # 日志与模型保存目录
│  ├─ __init__.py                   
│  ├─ RGBCamera.py                   # RGB 相机测试脚本
│  └─ train.py                       # 训练入口脚本
│
├─ Env_Task4/                        # 任务4：强化学习训练 + 开阔水域场景
│  ├─ env/                           
│  │  ├─ __init__.py                 
│  │  ├─ env_config.py               # 环境配置文件
│  │  ├─ openwater_env.py            # OpenWater 场景环境封装
│  │  ├─ rl_config.py                # 强化学习超参数配置
│  │  ├─ task4.py                    # 任务4环境实现
│  │  └─ trainer.py                  # 训练封装
│  ├─ logs/                          # 日志目录
│  ├─ logs1/                         # 模型、日志、评估结果目录
│  ├─ __init__.py                    
│  ├─ eval_one_episode.csv           
│  ├─ evaluate.py                    # 模型评估脚本
│  └─ train.py                       # 训练入口脚本
│
├─ LLM_Task/                         # 大语言模型任务控制模块
│  ├─ env/                           # 预留环境相关目录
│  ├─ logs/                          # 运行日志目录
│  ├─ __init__.py                    
│  ├─ llm_client.py                  # 大模型接口封装
│  ├─ llm_command_parser.py          # 自然语言指令解析
│  ├─ main.py                        # LLM 控制主入口
│  └─ task_dispatcher.py             # 任务分发逻辑
│
└─ stable_baselines3/                # 本地强化学习算法源码目录
  
