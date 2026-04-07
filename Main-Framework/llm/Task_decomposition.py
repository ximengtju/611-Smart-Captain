from modelscope import AutoModelForCausalLM, AutoTokenizer
import re


class TaskDecomposer:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        )

    def decompose_task(self, main_task, max_retries=3):
        """将主任务分解为多个子任务"""
        # 构建提示词
        instruction = "请将以下总任务分解为多个明确的子任务，每个子任务需有清晰的目标。"
        prompt = f"{instruction}\n{main_task}"

        messages = [
            {"role": "system", "content": "你是一个任务分解专家，擅长将复杂任务拆解为可执行的子任务。"},
            {"role": "user", "content": prompt}
        ]

        # 格式化输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 生成响应
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,  # 控制生成随机性
            do_sample=True
        )

        # 处理输出
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 解析子任务
        return self._parse_subtasks(response, max_retries)

    def parameter_generation(self, task_params, keys, max_retries=3):
        """将现有参数提炼为输入参数"""
        # 修复字符串格式化错误
        instruction = f'以下是这个智能体可能需要的全部有关参数，该智能体执行的为"{keys}"任务，请筛选出归纳出相关的输入参数'
        prompt = f"{instruction}\n{task_params}"

        messages = [
            {"role": "system", "content": "你是一个参数筛选专家，擅长将可用参数筛选出来作为输入参数。"},
            {"role": "user", "content": prompt}
        ]

        # 格式化输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 生成响应
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,  # 控制生成随机性
            do_sample=True
        )

        # 处理输出
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        response = self._parse_subtasks(response, max_retries)

        params_names, params_values = self.split_params(response)

        # 解析子任务
        return params_names, params_values

    def _parse_subtasks(self, response, max_retries):
        """解析模型响应，提取子任务列表"""
        if max_retries <= 0:
            raise RuntimeError("Max retries reached, failed to parse subtasks.")

        try:
            # 清理响应文本，移除可能的额外说明
            cleaned_response = response.strip()

            # 查找分号分隔的子任务
            subtasks = []
            if ';' in cleaned_response:
                # 按分号分割
                parts = cleaned_response.split(';')
                for part in parts:
                    part = part.strip()
                    if part and '-' in part:  # 确保有分隔符
                        subtasks.append(part)

            # 如果没有找到分号分隔的任务，尝试其他方法
            if not subtasks:
                # 尝试查找编号模式
                pattern = r'【\d+】(.+?)(?=【\d+】|$)'
                matches = re.findall(pattern, cleaned_response)
                if matches:
                    subtasks = [match.strip() for match in matches if match.strip()]

            # 如果还是没有找到，尝试按行解析
            if not subtasks:
                lines = cleaned_response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and '-' in line and not line.startswith(('请', '总任务', '注：')):
                        subtasks.append(line)

            # 验证至少有一个子任务
            if not subtasks:
                raise ValueError("No valid subtasks found in response")

            return subtasks

        except Exception as e:
            print(f"Error parsing subtasks: {e}. Retrying... ({max_retries} retries left)")
            return self._parse_subtasks(response, max_retries - 1)

    def split_params(self, response):
        params_names = []
        params_values = []

        for param in response:
            if ':' in param:
                parts = param.split(':', -1)
                params_names.append(parts[0].strip())
                params_values.append(parts[1].strip())
            else:
                params_names.append(param.strip())
                params_values.append("")

        return params_names, params_values

    def split_subtasks(self, subtasks):
        """将子任务列表分割为任务名称和任务描述两个数组"""
        task_names = []
        task_descriptions = []

        for task in subtasks:
            # 使用"-"分割任务名称和描述
            if '-' in task:
                parts = task.split('-', 1)  # 只分割第一个"-"
                task_names.append(parts[0].strip())
                task_descriptions.append(parts[1].strip())
            else:
                # 如果没有"-"，则将整个字符串作为任务名称，描述为空
                task_names.append(task.strip())
                task_descriptions.append("")

        return task_names, task_descriptions