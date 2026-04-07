import json
import os
from typing import List, Tuple, Dict, Optional, Union
from Task_decomposition import TaskDecomposer

class LLMInterface:
    """
    大语言模型接口：将自然语言任务分解为子任务并分类，最终映射到环境模式。
    """

    def __init__(
        self,
        llm_path: str,
        env=None,                               # 环境对象（需包含 set_multi_mode_index 方法）
        agents=None,                            # 代理对象（需包含 set_multi_mode_index 方法）
        mode_mapping: Optional[Dict[str, int]] = None,  # 任务类型 -> mode 索引映射
    ):

        self.llm = TaskDecomposer(llm_path)

        # 定义所有可用的任务类型（顺序即为默认 mode 索引）
        self.agent_use = [
            "导航", "目标跟踪", "区域躲避", "区域巡航", "路径跟踪",
            "定点悬停", "姿态控制", "环境检测", "地形测绘", "搜索", "记录", "无相应功能"
        ]

        self.env = env
        self.agents = agents

        # 建立任务类型到 mode 索引的映射
        if mode_mapping is None:
            self.mode_mapping = {task: idx for idx, task in enumerate(self.agent_use)}
        else:
            self.mode_mapping = mode_mapping

        self.last_subtasks = None          # 缓存最近一次分解的子任务
        self.last_classifications = None   # 缓存最近一次分类结果

    def match_agents(self, main_task: str, max_retries: int = 3) -> Tuple[List[str], List[str], List[str]]:

        # 分解任务
        subtasks = self.llm.decompose_task(main_task, max_retries)
        task_names, task_descriptions = self.llm.split_subtasks(subtasks)

        classifications = task_names

        self.last_subtasks = (task_names, task_descriptions)
        self.last_classifications = classifications

        return task_names, task_descriptions, classifications

    def set_mode_from_task(self, main_task: str, max_retries: int = 3) -> List[int]:
        # 分解并分类
        _, _, classifications = self.match_agents(main_task, max_retries)

        # 将分类映射为 mode 索引
        mode_indices = []
        for cls in classifications:
            if cls in self.mode_mapping:
                mode_idx = self.mode_mapping[cls]
            else:
                print(f"警告：未找到任务类型 '{cls}' 的 mode 映射，使用默认 0")
                mode_idx = 0
            mode_indices.append(mode_idx)

        # 如果有多个子任务，可以选择只使用第一个（或全部，取决于需求）
        # 这里选择使用第一个子任务对应的 mode（简单场景）
        if mode_indices:
            target_mode = mode_indices[0]
            print(f"设置环境模式为: {target_mode} (对应任务: {classifications[0]})")

            # 设置环境模式
            if self.env is not None and hasattr(self.env, 'set_multi_mode_index'):
                self.env.set_multi_mode_index(target_mode)
            else:
                print("警告：环境对象未提供 set_multi_mode_index 方法")

            # 设置代理模式
            if self.agents is not None and hasattr(self.agents, 'set_multi_mode_index'):
                self.agents.set_multi_mode_index(target_mode)
            else:
                print("警告：代理对象未提供 set_multi_mode_index 方法")

        return mode_indices

    # 以下为可选扩展方法，便于调试或后续使用

    def get_last_subtasks(self) -> Optional[Tuple[List[str], List[str]]]:
        return self.last_subtasks

    def get_last_classifications(self) -> Optional[List[str]]:
        return self.last_classifications

    def effect_test(self, json_path: str, res_path: str, text_num: int) -> None:
        with open(json_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        results = []

        for idx, item in enumerate(data):
            if idx >= text_num:
                break
            instruction = item["instruction"]
            inputs = item["input"]
            main_task = f"{instruction}{inputs}"

            _, _, classifications = self.match_agents(main_task)
            results.append({
                "input": inputs,
                "agent_keys": classifications,
                "task_name": results[-1][0] if results else [],
                "task_descriptions": results[-1][1] if results else [],
            })

        # 保存结果
        base_name = res_path.rsplit('.', 1)[0]
        json_file = f"{base_name}.json"
        with open(json_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"测试结果已保存到 {json_file}")