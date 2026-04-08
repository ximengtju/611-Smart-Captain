# llm_client.py
import os
from openai import OpenAI


class QwenClient:
    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: str = "自己的api",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("未找到 API Key，请设置 DASHSCOPE_API_KEY 环境变量。")

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def chat_json(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content