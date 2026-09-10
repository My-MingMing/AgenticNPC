"""
模型配置工具类
封装LLM API调用的配置管理（模型名称、API地址、密钥、温度等参数）
"""

import os
from dotenv import load_dotenv
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

load_dotenv()
load_dotenv(".local.env", override=True)


class ModelConfiguration(BaseModel):
    """Agent使用模型的配置类。"""
    model_name: Optional[str] = Field(default=None, description="代理使用的模型名称。")
    api_base: Optional[str] = Field(default=None, description="可选的 API 地址。")
    api_key: Optional[str] = Field(default=None, description="可选的 API 密钥。")
    temperature: float = Field(default=0.4, description="模型的温度参数。")
    max_tokens: int = Field(default=512, description="单次模型生成的最大 token 数。NPC 默认不需要长回复。")
    enable_thinking: bool = Field(default=False, description="是否启用有限思维模式。")
    thinking_budget: int = Field(default=256, description="思维模式预算，单位 token。")
    final_answer_max_tokens: int = Field(default=128, description="玩家可见最终回答的最大 token 数。")

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "ModelConfiguration":
        """从 RunnableConfig 创建一个 ModelConfiguration 实例。"""
        configurable: dict[str, Any] = config.get("configurable", {}) if config else {}

        # 兼容 Pydantic v1/v2
        fields = cls.model_fields if hasattr(cls, "model_fields") else cls.__fields__
        raw_values: Dict[str, Any] = {
            name: configurable.get(name, getattr(field, "default", None))
            for name, field in fields.items()
        }

        values = {k: v for k, v in raw_values.items() if v is not None}

        model_name = values.get("model_name")
        if not model_name:
            raise ValueError(f"model_name is required in config but got: {configurable}")

        api_configs = configurable.get("api_configs", {})
        model_api_config = api_configs.get(model_name, api_configs.get("default", {}))

        if "api_base" not in values and "api_base" in model_api_config:
            raw = model_api_config["api_base"]
            # 如果值本身是URL则直接使用，否则当作环境变量名查找
            if raw.startswith("http"):
                values["api_base"] = raw
            else:
                values["api_base"] = os.getenv(raw, raw)
        if "api_key" not in values and "api_key_env" in model_api_config:
            raw = model_api_config["api_key_env"]
            # 如果值本身是key则直接使用，否则当作环境变量名查找
            if raw.startswith("sk-") or raw.startswith("key-"):
                values["api_key"] = raw
            else:
                values["api_key"] = os.getenv(raw, raw)

        return cls(**values)
