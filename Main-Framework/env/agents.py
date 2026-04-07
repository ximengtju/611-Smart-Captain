import os
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

class Agents:
    def __init__(self, 
                 model_paths: Union[str, List[str]], 
                 model_types: Union[str, List[str]], 
                 env, 
                 model_class_dict: Dict[str, Any],
                 mode: str = 'single'):
        
        if isinstance(model_paths, str):
            model_paths = [model_paths]
            model_types = [model_types] if isinstance(model_types, str) else model_types

        self.model_class_dict = model_class_dict
        self.mode = mode
        self.models = []
        self.mode_index = 0  # 默认使用第一个模型
        self.mode_to_model_map = {} # 可选的自定义模式映射

        for path, typ in zip(model_paths, model_types):
            if typ not in model_class_dict:
                raise ValueError(f"不支持的模型类型: {typ}")
            model_class = model_class_dict[typ]
            model = model_class.load(path)
            self.models.append(model)
        
        if self.mode == 'single' and len(self.models) != 1:
            print("警告: 单模型模式下提供了多个模型，将仅使用第一个模型。")


    def set_mode(self, mode: str):
        self.mode = mode 


    def set_multi_mode_index(self, index: int):
        if index < 0 or index >= len(self.models):
            raise IndexError(f"无效模型索引 {index}，可用模型数量: {len(self.models)}")
        self.current_multi_mode_index = index


    def set_multi_mode_by_key(self, key: str):
        if key not in self.mode_to_model_map:
            raise KeyError(f"未找到模式映射键: {key}")
        self.current_multi_mode_index = self.mode_to_model_map[key]


    def add_mode_mapping(self, key: str, model_index: int):
        if model_index < 0 or model_index >= len(self.models):
            raise IndexError(f"无效的模型索引 {model_index}，可用模型数量: {len(self.models)}")
        self.mode_to_model_map[key] = model_index


    def predict(self, obs, state: Optional[Any] = None, deterministic: bool = True):
        if self.mode == 'single':
            model = self.models[0]
        else:  # multi mode
            model = self.models[self.current_multi_mode_index]
        
        action, new_state = model.predict(obs, state=state, deterministic=deterministic)
        return action, new_state
    

    @staticmethod
    def find_latest_model(env_name: str, model_type: str, log_dir: str = "./logs") -> Optional[str]:
        """
        寻找最新训练完毕的模型
        """
        log_dir_path = Path(log_dir)
        # 查找符合条件的目录
        matching_dirs = [d for d in log_dir_path.iterdir() if d.is_dir() and 
                         str(d.name).startswith(f"{env_name}_{model_type}_")]
        if not matching_dirs:
            print(f"未找到环境为 {env_name} 模型为 {model_type} 的训练日志")
            return None
        
        # 按照文件夹名称中的数字排序，找到最新的目录
        latest_dir = sorted(matching_dirs, key=lambda d: int(str(d.name).split("_")[-1]))[-1]
        
        # 查找该目录下的最新模型文件（使用zip文件）
        model_files = list(latest_dir.glob("*.zip"))
        if not model_files:
            print(f"在目录 {latest_dir} 中未找到zip格式的模型文件")
            return None
        
        # 如果模型文件名包含数字步骤，按步骤数排序找到最新的
        try:
            latest_model = sorted(model_files, 
                               key=lambda f: int(''.join(filter(str.isdigit, f.stem))))[-1]
        except:
            # 如果解析失败，则按修改时间排序
            latest_model = sorted(model_files, key=os.path.getmtime)[-1]
        
        print(f"找到最新模型: {latest_model}")
        return str(latest_model)