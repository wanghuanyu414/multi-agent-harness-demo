#!/usr/bin/env python3
"""旅行助手 Web 服务 —— 最小改动，复用 init_system()"""
import logging
import os
import sys
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from main import load_config, setup_logging, init_system, request_id_var

logger = logging.getLogger("web")


class SlidingWindowRateLimiter:
    """进程内滑动窗口限流。单进程够用；多进程部署需换 Redis 后端。"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        dq = self._hits[key]
        while dq and now - dq[0] > self.window_seconds:
            dq.popleft()
        if len(dq) >= self.max_requests:
            return False
        dq.append(now)
        return True

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = Field(default=None, description="会话 ID；不传则由服务端生成并返回，客户端应保存复用")

class ChatResponse(BaseModel):
    response: str
    thread_id: str | None = None       # 会话 ID（服务端生成时回传，供多轮复用）
    intent: str | None = None
    guardrail: dict | None = None     # 护栏检测结果
    trace: dict | None = None          # trace 汇总（仅 ?trace=true 时返回）
    agents: list[str] | None = None   # 多 Agent 协作链路
    reasoning: str | None = None      # 规划推理（CoT）
    human_review: dict | None = None  # 待人工审核信息（非空表示需要人工介入）


class ReviewRequest(BaseModel):
    thread_id: str = Field(..., description="会话 ID，必须与 chat 返回的一致，用于 resume")
    approve: bool = Field(default=True, description="是否通过审核")
    response: str | None = Field(default=None, description="编辑后的回复内容（通过时可选）")


def _build_chat_response(result: dict) -> ChatResponse:
    """根据 graph 执行结果构建标准 ChatResponse（非中断路径）"""
    span_chain = result.get("span_chain", [])
    if span_chain:
        from agent.harness.observability import emit_span_chain_summary
        emit_span_chain_summary(span_chain, color=True)

    guard_result = result.get("input_guardrail_result") or result.get("output_guardrail_result")
    guard_info = None
    if guard_result:
        guard_info = {
            "passed": guard_result.get("passed"),
            "mode": guard_result.get("mode"),
            "violations": guard_result.get("violations", []),
        }

    trace_info = None
    if span_chain:
        trace_info = {
            "total_ms": round(sum(s["duration_ms"] for s in span_chain), 2),
            "nodes": len(span_chain),
            "errors": [s for s in span_chain if s["error"]],
            "spans": [{"node": s["node"], "duration_ms": s["duration_ms"], "error": s["error"]} for s in span_chain],
        }

    return ChatResponse(
        response=result.get("final_response", ""),
        intent=result.get("intent"),
        guardrail=guard_info,
        trace=trace_info,
        agents=result.get("agent_trace") or None,
        reasoning=result.get("plan_reasoning") or None,
    )

# ---------------------------------------------------------------------------
# Lifespan: start MCP once, keep alive across requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level)
    graph, mcp_client, close_checkpointer = await init_system(config)
    app.state.graph = graph
    app.state.mcp_client = mcp_client
    logger.info("旅行助手 Web 服务就绪")
    yield
    logger.info("正在关闭 MCP 连接...")
    await mcp_client.close_all()
    if close_checkpointer:
        await close_checkpointer()
    logger.info("服务已关闭")

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.abspath(__file__))


def create_app() -> FastAPI:
    app = FastAPI(title="旅行助手", lifespan=lifespan)

    # 请求限流（按客户端 IP 滑动窗口）
    rl = load_config().get("rate_limit", {})
    rl_enabled = bool(rl.get("enabled", True))
    limiter = SlidingWindowRateLimiter(
        max_requests=int(rl.get("max_requests", 30)),
        window_seconds=int(rl.get("window_seconds", 60)),
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if rl_enabled and request.url.path in ("/api/chat", "/api/review"):
            ip = request.client.host if request.client else "unknown"
            if not limiter.allow(ip):
                return JSONResponse(status_code=429, content={"detail": "请求过于频繁，请稍后再试"})
        return await call_next(request)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        # 后注册 = 最外层，确保所有响应（含 429）都带 X-Request-ID
        rid = uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_var.reset(token)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        mcp = getattr(request.app.state, "mcp_client", None)
        if mcp is None:
            return JSONResponse(status_code=503, content={"status": "not_ready", "reason": "MCP 未初始化"})
        dead = mcp.unhealthy_servers()
        if dead:
            return JSONResponse(status_code=503, content={"status": "not_ready", "reason": f"MCP 服务已断开: {dead}"})
        return {"status": "ready"}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request):
        # 未传 thread_id 时生成独立会话
        tid = req.thread_id or str(uuid.uuid4())
        try:
            result = await request.app.state.graph.ainvoke(
                {"messages": [HumanMessage(content=req.message)]},
                {"configurable": {"thread_id": tid}},
            )

            # 人工审核中断：返回待审信息，等待前端提交审核决策后 resume
            interrupts = result.get("__interrupt__")
            if interrupts:
                payload = interrupts[0].value if interrupts else {}
                if isinstance(payload, dict) and payload.get("type") == "human_review":
                    return ChatResponse(
                        response="",
                        thread_id=tid,
                        intent=result.get("intent"),
                        human_review={
                            "intent": payload.get("intent"),
                            "response": payload.get("response"),
                        },
                    )
                logger.warning(f"未知中断类型，返回提示: {payload}")
                return ChatResponse(response="系统已暂停等待人工处理，但无法识别中断类型。", thread_id=tid)

            resp = _build_chat_response(result)
            resp.thread_id = tid
            return resp
        except Exception:
            logger.exception("Chat 处理失败 thread_id=%s", tid)
            return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请稍后重试"})

    @app.post("/api/review", response_model=ChatResponse)
    async def review(req: ReviewRequest, request: Request):
        try:
            decision = {"approve": req.approve}
            if req.response:
                decision["response"] = req.response
            result = await request.app.state.graph.ainvoke(
                Command(resume=decision),
                {"configurable": {"thread_id": req.thread_id}},
            )
            resp = _build_chat_response(result)
            resp.thread_id = req.thread_id
            return resp
        except Exception:
            logger.exception("Review 处理失败 thread_id=%s", req.thread_id)
            return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请稍后重试"})

    static_dir = os.path.join(_ROOT, "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    return app

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    import asyncio
    app = create_app()
    port = int(os.environ.get("PORT", "8080"))
    config = uvicorn.Config(app, host="127.0.0.1", port=port)
    server = uvicorn.Server(config)
    try:
        asyncio.run(server.serve())
    except TypeError:
        # PyCharm 调试器 patch 了 asyncio.run()，不支持 loop_factory 参数
        # 回退到手动管理事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
