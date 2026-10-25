"""评估与性能度量 —— Token 计量 / 节点延迟 / 工具调用遥测 / 百分位统计

为评估体系与性能治理提供量化基线：
- count_tokens / count_message_tokens：Token 估算（tiktoken 优先，启发式兜底）
- MetricsCollector：进程内单例，收集节点延迟、工具调用成功率/耗时/token、累计 token 用量
- percentiles：p50/p95/p99 聚合
"""
import contextvars
import logging
import re
import threading
import time
from collections import defaultdict

logger = logging.getLogger("metrics")

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # 无 langchain 时退化，避免硬依赖
    BaseCallbackHandler = object

try:
    from langchain.agents.middleware import AgentMiddleware
except Exception:  # 无 langchain 1.x 时退化，避免硬依赖
    AgentMiddleware = object

_CJK_RE = re.compile(r"[一-鿿]")


def count_tokens(text) -> int:
    """估算文本 token 数。tiktoken 优先；兜底用中文按字、ASCII 按 4 字符 ≈ 1 token。"""
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(str(text)))
        except Exception:
            pass
    s = str(text)
    cjk = len(_CJK_RE.findall(s))
    other = len(s) - cjk
    return cjk + (other // 4) + (1 if other % 4 else 0)


def count_message_tokens(messages) -> int:
    """估算消息列表总 token 数（每条消息另计 ~4 token 角色/格式开销）"""
    total = 0
    for m in messages:
        c = m.content if hasattr(m, "content") else str(m)
        total += count_tokens(c if isinstance(c, str) else str(c)) + 4
    return total


def percentiles(values, ps=(50, 95, 99)) -> dict:
    """返回 {p50: .., p95: .., p99: ..}；空列表对应 None。"""
    if not values:
        return {f"p{p}": None for p in ps}
    s = sorted(values)
    out = {}
    for p in ps:
        idx = min(len(s) - 1, int((p / 100.0) * (len(s) - 1)))
        out[f"p{p}"] = round(s[idx], 2)
    return out


class MetricsCollector:
    """进程内指标收集器（单例），供节点/工具/Eval 共用。"""

    _instance = None

    def __init__(self):
        self._lock = threading.Lock()
        self.node_durations: dict[str, list] = defaultdict(list)
        self.model_durations: dict[str, list] = defaultdict(list)
        self.tool_stats: dict[str, dict] = {}
        self.tokens: dict[str, int] = defaultdict(int)

    @classmethod
    def get(cls) -> "MetricsCollector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        with self._lock:
            self.node_durations.clear()
            self.model_durations.clear()
            self.tool_stats.clear()
            self.tokens.clear()

    # ---- 记录 ----
    def record_node(self, name: str, duration_ms: float) -> None:
        with self._lock:
            self.node_durations[name].append(duration_ms)

    def record_model(self, name: str, duration_ms: float) -> None:
        with self._lock:
            self.model_durations[name].append(duration_ms)

    def record_tool(self, name: str, success: bool, duration_ms: float, tokens: int = 0) -> None:
        with self._lock:
            st = self.tool_stats.setdefault(name, {
                "success": 0, "fail": 0, "durations": [], "tokens": [],
            })
            st["success" if success else "fail"] += 1
            st["durations"].append(duration_ms)
            st["tokens"].append(tokens)

    def add_tokens(self, input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0) -> None:
        with self._lock:
            self.tokens["input"] += input_tokens
            self.tokens["output"] += output_tokens
            self.tokens["cached"] += cached_tokens
            self.tokens["total"] += input_tokens + output_tokens

    # ---- 汇总 ----
    def node_latency_summary(self) -> dict:
        with self._lock:
            data = {name: list(ds) for name, ds in self.node_durations.items()}
        return {name: percentiles(ds) for name, ds in data.items()}

    def model_latency_summary(self) -> dict:
        with self._lock:
            data = {name: list(ds) for name, ds in self.model_durations.items()}
        return {name: percentiles(ds) for name, ds in data.items()}

    def tool_summary(self) -> dict:
        with self._lock:
            snapshot = {
                name: {
                    "success": st["success"], "fail": st["fail"],
                    "durations": list(st["durations"]), "tokens": list(st["tokens"]),
                }
                for name, st in self.tool_stats.items()
            }
        out = {}
        for name, st in snapshot.items():
            total = st["success"] + st["fail"]
            out[name] = {
                "calls": total,
                "success_rate": round(st["success"] / total, 3) if total else None,
                "latency": percentiles(st["durations"]),
                "tokens": percentiles(st["tokens"]) if any(st["tokens"]) else None,
            }
        return out

    def token_summary(self) -> dict:
        with self._lock:
            return dict(self.tokens)


# 请求级 token 追踪（contextvars 隔离并发请求，避免全局累计被并发互相污染）
_request_tokens: contextvars.ContextVar = contextvars.ContextVar("request_tokens", default=None)


def start_request_tokens() -> None:
    """请求开始：初始化当前请求（context）的 token 计数。"""
    _request_tokens.set({"input": 0, "output": 0, "cached": 0})


def add_request_tokens(input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0) -> None:
    """在请求级上下文累加 token（并发安全，仅影响当前请求 context）。"""
    tok = _request_tokens.get()
    if tok is not None:
        tok["input"] += input_tokens
        tok["output"] += output_tokens
        tok["cached"] += cached_tokens


def get_request_tokens() -> dict:
    """请求结束：返回本次请求的 token 用量（input/output/cached）。"""
    tok = _request_tokens.get()
    return dict(tok) if tok else {"input": 0, "output": 0, "cached": 0}


# 请求级 token 流式队列（contextvars，Worker 流式生成时写入，web 层读取）
_stream_queue: contextvars.ContextVar = contextvars.ContextVar("stream_queue", default=None)


def set_stream_queue(q) -> None:
    """设置当前请求的 token 流式队列。"""
    _stream_queue.set(q)


def get_stream_queue():
    """返回当前请求的 token 流式队列（未设置时为 None）。"""
    return _stream_queue.get()


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """LLM 回调：捕获每次调用的真实 token 用量（usage_metadata），写入 MetricsCollector。

    挂到 ChatOpenAI 构造参数 callbacks=[...] 上，所有调用点（含 model_copy 派生的
    supervisor / guardrail 判断）自动上报，无需逐处埋点。
    """

    def __init__(self):
        self.collector = MetricsCollector.get()
        self.calls = 0

    def on_llm_end(self, response, **kwargs) -> None:
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0

        def _cached(um: dict) -> int:
            # OpenAI 兼容端点：prompt_tokens_details / input_token_details 里带 cached_tokens
            for k in ("input_token_details", "prompt_tokens_details"):
                d = um.get(k)
                if isinstance(d, dict) and d.get("cached_tokens"):
                    return int(d["cached_tokens"])
            return 0

        llm_output = getattr(response, "llm_output", None) or {}
        tu = llm_output.get("token_usage") or {}
        if tu:
            input_tokens = tu.get("prompt_tokens") or tu.get("input_tokens") or 0
            output_tokens = tu.get("completion_tokens") or tu.get("output_tokens") or 0
            cached_tokens = _cached(tu)
        else:
            for gen_list in (getattr(response, "generations", None) or []):
                for g in gen_list:
                    msg = getattr(g, "message", None)
                    um = getattr(msg, "usage_metadata", None) or {}
                    if not um:
                        gi = getattr(g, "generation_info", None) or {}
                        um = gi.get("usage_metadata") or gi.get("token_usage") or {}
                    input_tokens += um.get("input_tokens") or um.get("prompt_tokens") or 0
                    output_tokens += um.get("output_tokens") or um.get("completion_tokens") or 0
                    cached_tokens += _cached(um)

        if input_tokens or output_tokens:
            self.collector.add_tokens(input_tokens, output_tokens, cached_tokens)
            add_request_tokens(input_tokens, output_tokens, cached_tokens)
            self.calls += 1


class ModelTimingMiddleware(AgentMiddleware):
    """Agent middleware：统计 create_agent 内部每次 LLM 模型调用耗时。

    挂在 create_agent 的 middleware 参数上，覆盖 supervisor/summarize 等
    手写节点无法用 middleware 计时的盲区（仅 worker 内部模型调用维度）。
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    @property
    def name(self) -> str:
        return f"{self.agent_name}.model_timing"

    async def awrap_model_call(self, request, handler):
        start = time.monotonic()
        try:
            return await handler(request)
        finally:
            MetricsCollector.get().record_model(
                self.agent_name, (time.monotonic() - start) * 1000
            )
