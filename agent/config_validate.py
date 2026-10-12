"""配置校验 —— 启动时校验配置，发现占位符 key / 非法值立即报错。"""
import logging

logger = logging.getLogger(__name__)

_PLACEHOLDER_SUBSTRINGS = (
    "sk-your-", "your-api-key", "changeme", "placeholder", "lsv2_pt_local",
)
_VALID_BACKENDS = {"redis", "sqlite", "postgres", "memory"}
_VALID_MODES = {"blocking", "warning"}


class ConfigError(ValueError):
    """配置不合法，启动应中止。"""


def _is_placeholder(value) -> bool:
    if not value or not isinstance(value, str):
        return True
    v = value.strip().lower()
    return any(p in v for p in _PLACEHOLDER_SUBSTRINGS)


def validate_config(config: dict) -> None:
    """校验关键配置，非法时抛出 ConfigError（附具体字段路径）。"""
    errors: list[str] = []

    llm = config.get("llm") or {}
    if _is_placeholder(llm.get("api_key")):
        errors.append("llm.api_key 缺失或为占位符，请在 .env 设置 OPENAI_API_KEY")

    cp = (config.get("agent") or {}).get("checkpoint") or {}
    backend = str(cp.get("backend", "sqlite")).lower()
    if backend not in _VALID_BACKENDS:
        errors.append(
            f"agent.checkpoint.backend 非法: {backend!r}，应为 {sorted(_VALID_BACKENDS)}"
        )

    guardrails = (config.get("harness") or {}).get("guardrails") or {}
    for side in ("input", "output"):
        g = guardrails.get(side) or {}
        mode = g.get("mode")
        if mode and mode not in _VALID_MODES:
            errors.append(
                f"harness.guardrails.{side}.mode 非法: {mode!r}，应为 {sorted(_VALID_MODES)}"
            )

    # LangSmith 占位符 key：不阻断启动，但明确告警（云端 trace/prompt 会静默失败）
    ls = config.get("langsmith") or {}
    if ls.get("enabled") and _is_placeholder(ls.get("api_key")):
        logger.warning("langsmith 已启用但 api_key 疑似占位符，云端 trace / prompt 拉取将静默失败")

    if errors:
        raise ConfigError("配置校验失败:\n- " + "\n- ".join(errors))
