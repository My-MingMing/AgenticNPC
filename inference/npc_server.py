"""
NPC对话推理服务
基于FastAPI的REST API，用于游戏引擎集成
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis


# ====== 数据模型 ======

class NPCInitRequest(BaseModel):
    session_id: str
    npc_id: str
    npc_policy: str
    world_state: Dict[str, Any]
    player_id: str


class NPCChatRequest(BaseModel):
    session_id: str
    player_message: str


class NPCChatResponse(BaseModel):
    npc_message: str
    tool_calls: List[Dict[str, Any]] = []
    state_changes: List[Dict[str, Any]] = []
    session_active: bool = True


class NPCResetRequest(BaseModel):
    session_id: str


# ====== 全局组件 ======

redis_client: Optional[redis.Redis] = None
llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global redis_client, llm_client
    from openai import AsyncOpenAI

    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )

    llm_client = AsyncOpenAI(
        base_url=os.getenv("NPC_MODEL_API_BASE", "http://localhost:8000/v1"),
        api_key=os.getenv("NPC_MODEL_API_KEY", "EMPTY"),
    )

    yield

    await redis_client.close()


app = FastAPI(title="NPC Dialogue Service", lifespan=lifespan)


# ====== API端点 ======

@app.post("/npc/init")
async def init_npc_session(req: NPCInitRequest):
    """初始化NPC对话会话"""
    session_data = {
        "npc_id": req.npc_id,
        "npc_policy": req.npc_policy,
        "world_state": json.dumps(req.world_state),
        "player_id": req.player_id,
        "history": json.dumps([]),
        "turn_count": "0",
    }

    await redis_client.hset(f"npc_session:{req.session_id}", mapping=session_data)
    await redis_client.expire(f"npc_session:{req.session_id}", 1800)  # 30min TTL

    return {"status": "ok", "session_id": req.session_id}


@app.post("/npc/chat", response_model=NPCChatResponse)
async def npc_chat(req: NPCChatRequest):
    """处理玩家消息，返回NPC回复"""
    session_key = f"npc_session:{req.session_id}"
    session = await redis_client.hgetall(session_key)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = json.loads(session["history"])
    world_state = json.loads(session["world_state"])
    npc_policy = session["npc_policy"]
    turn_count = int(session["turn_count"])

    # 构建消息
    messages = _build_messages(npc_policy, history, req.player_message, world_state)

    # 调用模型
    try:
        response = await llm_client.chat.completions.create(
            model=os.getenv("NPC_MODEL_NAME", "qwen3-8b-npc"),
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        npc_raw_response = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # 解析NPC回复（提取文本、工具调用、状态变更）
    npc_message, tool_calls, state_changes = _parse_npc_response(npc_raw_response, world_state)

    # 更新会话历史
    history.append({"role": "player", "content": req.player_message})
    history.append({"role": "npc", "content": npc_message, "tool_calls": tool_calls})

    # 判断会话是否结束
    session_active = "###STOP" not in npc_raw_response and turn_count < 20

    # 保存状态
    await redis_client.hset(session_key, mapping={
        "history": json.dumps(history),
        "world_state": json.dumps(world_state),
        "turn_count": str(turn_count + 1),
    })
    await redis_client.expire(session_key, 1800)

    return NPCChatResponse(
        npc_message=npc_message,
        tool_calls=tool_calls,
        state_changes=state_changes,
        session_active=session_active,
    )


@app.post("/npc/reset")
async def reset_session(req: NPCResetRequest):
    """重置对话会话"""
    await redis_client.delete(f"npc_session:{req.session_id}")
    return {"status": "ok"}


@app.get("/npc/status")
async def health_check():
    """健康检查"""
    try:
        await redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception:
        return {"status": "degraded", "redis": "disconnected"}


# ====== 内部函数 ======

def _build_messages(
    npc_policy: str,
    history: List[Dict],
    player_message: str,
    world_state: Dict,
) -> List[Dict[str, str]]:
    """构建发送给模型的消息列表"""
    messages = [
        {"role": "system", "content": npc_policy},
    ]

    # 添加历史对话
    for turn in history[-10:]:  # 只保留最近10轮
        if turn["role"] == "player":
            messages.append({"role": "user", "content": turn["content"]})
        elif turn["role"] == "npc":
            messages.append({"role": "assistant", "content": turn["content"]})

    # 添加当前玩家消息
    messages.append({"role": "user", "content": player_message})

    return messages


def _parse_npc_response(
    raw_response: str,
    world_state: Dict,
) -> tuple:
    """
    解析NPC的原始响应。
    提取：自然语言回复、工具调用、状态变更。
    """
    import re

    tool_calls = []
    state_changes = []
    npc_message = raw_response

    # 提取工具调用
    tool_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_matches = re.findall(tool_pattern, raw_response, re.DOTALL)

    for match in tool_matches:
        try:
            tool_data = json.loads(match.strip())
            tool_calls.append(tool_data)
        except json.JSONDecodeError:
            pass

    # 提取最终给玩家的文本回复（去除工具调用和内部标签）
    npc_message = re.sub(r'<tool_call>.*?</tool_call>', '', npc_message, flags=re.DOTALL)
    npc_message = re.sub(r'<tool_response>.*?</tool_response>', '', npc_message, flags=re.DOTALL)
    npc_message = re.sub(r'<think>.*?</think>', '', npc_message, flags=re.DOTALL)
    npc_message = npc_message.replace("###STOP", "").strip()

    # 如果有answer标签，提取其中内容
    answer_match = re.search(r'<answer>(.*?)</answer>', npc_message, re.DOTALL)
    if answer_match:
        npc_message = answer_match.group(1).strip()

    return npc_message, tool_calls, state_changes


# ====== 入口 ======

if __name__ == "__main__":
    import uvicorn

    test_mode = "--test_mode" in __import__("sys").argv

    if test_mode:
        print("Running in test mode...")
        # 简单测试逻辑
        print("NPC Server ready for testing.")
    else:
        uvicorn.run(
            app,
            host=os.getenv("NPC_SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("NPC_SERVER_PORT", 8080)),
        )
