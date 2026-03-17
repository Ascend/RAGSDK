#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import asyncio
import json
import random
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 初始化 FastAPI 应用
app = FastAPI(title="Mock Embedding API Server", version="1.0")

# 跨域配置（保留，前端调用必备）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="角色：user/assistant/system")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field("default-model", description="模型名称（可选）")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(1.0, ge=0, le=2)
    max_tokens: Optional[int] = Field(1024, ge=1)
    stream: Optional[bool] = Field(False)


class EmbeddingRequest(BaseModel):
    model: Optional[str] = Field("default-embed-model", description="模型名称（可选）")
    input: List[str] | str  # 支持单文本/文本列表
    encoding_format: Optional[str] = Field("float")


class RerankRequest(BaseModel):
    query: str
    texts: List[str]


class ClipEmbeddingRequest(BaseModel):
    data: list
    parameters: dict


def generate_mock_chat_response(messages: List[ChatMessage]) -> str:
    last_user_msg = [msg for msg in messages if msg.role == "user"][-1].content
    return f"模拟回复：{last_user_msg} 的相关内容..."


def generate_mock_embedding(text: str) -> List[float]:
    random.seed(hash(text))  # 相同文本返回相同向量
    return [random.uniform(-1.0, 1.0) for _ in range(1024)]  # 改为1024维


def assemble_completion(request, case):
    content = generate_mock_chat_response(request.messages)
    if case == "qa_generate":
        content = "Q1：2024年高考语文作文题目是什么？\n参考段落：新课标Ⅰ卷。"
    result = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(asyncio.get_event_loop().time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": random.randint(10, 100), "completion_tokens": random.randint(10, 200),
                  "total_tokens": random.randint(20, 300)}
    }
    if case == "stream":
        results = []
        for i, content_char in enumerate(content):
            reason = "stop" if i == len(content) - 1 else None
            result = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": request.model,
                "choices": [{
                    "index": i,
                    "delta": {"role": "assistant", "content": content_char},
                    "finish_reason": reason
                }],
                "usage": {"prompt_tokens": random.randint(10, 100), "completion_tokens": random.randint(10, 200),
                          "total_tokens": random.randint(20, 300)}
            }
            results.append(result)
        return results
    return result


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    await asyncio.sleep(random.uniform(0.1, 1.0))
    return assemble_completion(request, "common")


@app.post("/v1/chat/completions_qa_generate")
async def completions_qa_generate(request: ChatCompletionRequest):
    return assemble_completion(request, "qa_generate")


@app.post("/v1/chat/completions_stream")
async def completions_stream(request: ChatCompletionRequest):
    results = assemble_completion(request, "stream")

    async def generate():
        for result in results:
            await asyncio.sleep(0.1)
            yield json.dumps(result) + "\n"

    # header修改无法生效
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    await asyncio.sleep(random.uniform(0.05, 0.5))

    # 处理输入
    texts = [request.input] if isinstance(request.input, str) else request.input
    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="输入文本不能为空")

    # 生成1024维向量
    data = []
    total_tokens = 0
    for idx, text in enumerate(texts):
        embedding = generate_mock_embedding(text)  # 固定1024维
        tokens = len(text.split())
        total_tokens += tokens
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": embedding,
        })

    return {
        "object": "list",
        "data": data,
        "model": request.model,  # 可选参数，不影响使用
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens}
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/rerank")
async def rerank(request: RerankRequest):
    query = request.query
    texts = request.texts
    if not (isinstance(texts, list) and len(texts) > 0):
        raise HTTPException(status_code=400, detail="`texts` cannot be empty")
    if not (isinstance(query, str) and len(query) > 0):
        raise HTTPException(status_code=400, detail="`query` cannot be empty")
    results = []
    for idx, text in enumerate(texts):
        random.seed(hash(text))
        score = random.uniform(0.0, 1.0)
        results.append({
            "index": idx,
            "score": score,
        })
    return results


@app.post("/encode_clip")
async def encode_clip(request: ClipEmbeddingRequest):
    datas = request.data
    results = []
    for idx, data in enumerate(datas):
        random.seed(hash(data.get("text")))
        results.append({"embedding": [random.uniform(-1.0, 1.0) for _ in range(1024)]})
    return {"data": results}


@app.post("/text2img")
async def text2img():
    return "base64 mock image"


# 启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
