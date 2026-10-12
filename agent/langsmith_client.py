"""LangSmith 集成 —— SaaS 客户端、节点追踪包装、prompt 远程化

与自研 harness.observability（控制台 span）并存：LangSmith 负责云端 trace /
评测 / prompt 版本管理，本地控制台输出保持不变。未启用时所有函数退化为直通，
不影响主流程。

环境变量经 langsmith.utils.get_env_var 读取，命名空间优先级 LANGSMITH_* > LANGCHAIN_*。
注意 get_env_var 有 lru_cache，故 init_langsmith 必须在首次 LLM 调用前执行。
"""
import logging
import os
from typing import Any, Callable

logger = logging.getLogger("agent.langsmith")

# prompt 在 LangSmith Hub 里的命名空间（push / pull 共用，保证标识符一致）
PROMPT_NAMESPACE = "travel-agent"

_enabled = False
_client: Any = None
_client_initialized = False


def init_langsmith(cfg: dict | None = None) -> bool:
    """按配置初始化 LangSmith，必须在首次 LLM 调用前执行。返回是否启用。

    Args:
        cfg: config.yaml 里的 langsmith 段 {"enabled": bool, "endpoint": str,
             "api_key": str, "project": str}
    """
    global _enabled
    cfg = cfg or {}
    _enabled = bool(cfg.get("enabled", False))

    if _enabled:
        endpoint = str(cfg.get("endpoint", "https://api.smith.langchain.com"))
        api_key = str(cfg.get("api_key", "lsv2_pt_local"))
        project = str(cfg.get("project", "travel-agent"))

        # 占位符 key 检测
        if (not api_key or api_key == "lsv2_pt_local"
                or api_key.startswith(("sk-your-", "your-api-key", "changeme", "placeholder"))):
            logger.warning(
                "LangSmith 已启用但 API Key 疑似占位符（%s），云端 trace / prompt 拉取将静默失败；"
                "请在 .env 设置真实 LANGSMITH_API_KEY", api_key)

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = project
        # 兼容旧版 langchain-core 仍读 LANGCHAIN_* 的场景
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
        logger.info("LangSmith 已启用 (endpoint=%s, project=%s)", endpoint, project)
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.info("LangSmith 未启用")

    return _enabled


def is_enabled() -> bool:
    """LangSmith 是否已启用"""
    return _enabled


def get_client():
    """返回 LangSmith Client 单例；未启用或初始化失败返回 None。"""
    global _client, _client_initialized
    if not _enabled:
        return None
    if not _client_initialized:
        _client_initialized = True
        try:
            from langsmith import Client
            _client = Client()
        except Exception as e:
            logger.warning("LangSmith Client 初始化失败: %s", e)
            _client = None
    return _client


def traceable_node(name: str):
    """给 graph 节点函数加 LangSmith span（与自研 trace_node 并存）。

    返回装饰器；未启用时原样返回函数。
    """
    def decorator(fn: Callable) -> Callable:
        if not _enabled:
            return fn
        try:
            from langsmith import traceable
            return traceable(name=name, run_type="chain")(fn)
        except Exception as e:
            logger.warning("traceable 包装节点 %s 失败: %s", name, e)
            return fn
    return decorator


def get_prompt(name: str, fallback: str) -> str:
    """从 LangSmith Hub 拉取 prompt 模板字符串；失败或未启用回退到本地 fallback。

    返回原始模板字符串（`{}`/`{{}}` 占位符语义与本地常量一致），
    调用方原有的 .format() / .replace() 逻辑保持不变。
    """
    client = get_client()
    if client is None:
        return fallback
    try:
        obj = client.pull_prompt(f"{PROMPT_NAMESPACE}/{name}")
        template = _prompt_to_string(obj)
        if template:
            logger.info("prompt '%s' 命中 Hub", name)
            return template
    except Exception as e:
        logger.debug("prompt '%s' 从 Hub 拉取失败，回退本地: %s", name, e)
    return fallback


def _prompt_to_string(prompt: Any) -> str:
    """从 pull 回的 LangChain prompt 对象提取原始模板字符串。"""
    if prompt is None:
        return ""
    # ChatPromptTemplate：messages 里每个消息带 .prompt.template
    msgs = getattr(prompt, "messages", None)
    if msgs:
        parts = []
        for m in msgs:
            inner = getattr(m, "prompt", None)
            t = getattr(inner, "template", None)
            if isinstance(t, str):
                parts.append(t)
        if parts:
            return parts[0] if len(parts) == 1 else "\n\n".join(parts)
    # PromptTemplate：直接 .template
    t = getattr(prompt, "template", None)
    if isinstance(t, str):
        return t
    return str(prompt)
