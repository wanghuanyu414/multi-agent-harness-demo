"""可观测性 —— 结构化追踪装饰器和 span 输出"""
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from ..metrics import MetricsCollector

logger = logging.getLogger("harness.observability")

_NODE_COLORS: dict[str, str] = {
    "analyze_intent": "CYAN",
    "input_guardrail": "YELLOW",
    "route_tool": "MAGENTA",
    "plan_execution": "MAGENTA",
    "execute_tool": "BLUE",
    "reflect": "YELLOW",
    "generate_response": "GREEN",
    "output_guardrail": "YELLOW",
}


def trace_node(node_name: str, config: dict):
    """装饰器工厂：为 async 节点函数添加可观测性追踪

    Args:
        node_name: 节点名称
        config: observability 配置 {"level": "basic|full", "output": "console|json|both", "color": bool}

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(state: dict) -> dict[str, Any]:
            from .otel import get_tracer
            tracer = get_tracer()
            with tracer.start_as_current_span(node_name) as otel_span:
                trace_id = state.get("trace_id", "")
                if not trace_id:
                    trace_id = uuid.uuid4().hex[:12]

                # 捕获输入 state key（非空值）
                input_keys = [k for k in state.keys() if state.get(k) not in ("", 0, False, [], {}, None)]

                start = time.monotonic()
                error = None
                result: dict[str, Any] = {}

                try:
                    raw = func(state)
                    # 支持同步和异步两种节点
                    import asyncio
                    if asyncio.iscoroutine(raw):
                        result = await raw
                    else:
                        result = raw
                except Exception as e:
                    error = type(e).__name__
                    otel_span.record_exception(e)
                    raise
                finally:
                    duration_ms = (time.monotonic() - start) * 1000
                    otel_span.set_attribute("node.name", node_name)
                    otel_span.set_attribute("duration_ms", round(duration_ms, 2))
                    if error:
                        otel_span.set_attribute("error", True)
                    MetricsCollector.get().record_node(node_name, duration_ms)
                    # 节点可能返回 Command（如 human_review 的 interrupt/resume），非 dict 时只发 span、不回写
                    is_dict = isinstance(result, dict)
                    span = {
                        "trace_id": trace_id,
                        "node": node_name,
                        "duration_ms": round(duration_ms, 2),
                        "input_state_keys": input_keys,
                        "output_state_keys": sorted(result.keys()) if is_dict and result else [],
                        "error": error,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    _emit_span(span, config)
                    if is_dict:
                        chain = list(state.get("span_chain", []))
                        chain.append(span)
                        result["span_chain"] = chain
                        result["trace_id"] = trace_id

                return result
        return wrapper
    return decorator


def _emit_span(span: dict, config: dict):
    """根据配置输出单个 span"""
    enabled = config.get("enabled", True)
    if not enabled:
        return
    level = config.get("level", "basic")
    output = config.get("output", "console")
    color_enabled = config.get("color", True)

    msg = _format_basic(span) if level == "basic" else json.dumps(span, ensure_ascii=False)

    if output in ("console", "both"):
        _console_out(msg, color_enabled, span.get("error"))

    if output in ("json", "both"):
        logger.info(json.dumps(span, ensure_ascii=False))


def _format_basic(span: dict) -> str:
    node = span["node"]
    duration = span["duration_ms"]
    status = "ERROR" if span["error"] else "OK"
    inputs = ",".join(span.get("input_state_keys", [])[:5])
    outputs = ",".join(span.get("output_state_keys", [])[:5])
    return f"[trace] {node} ({duration:.1f}ms) [{status}] in=({inputs}) out=({outputs})"


def _console_out(msg: str, color: bool, is_error: bool):
    if color:
        try:
            from colorama import Fore, Style
            if is_error:
                print(f"{Fore.RED}{msg}{Style.RESET_ALL}", file=sys.stderr)
            else:
                print(f"{Fore.CYAN}{msg}{Style.RESET_ALL}", file=sys.stderr)
            return
        except ImportError:
            pass
    print(msg, file=sys.stderr)


def emit_span_chain_summary(span_chain: list[dict], color: bool = True):
    """在所有 span 完成后打印汇总"""
    if not span_chain:
        return

    total_ms = sum(s["duration_ms"] for s in span_chain)
    errors = [s for s in span_chain if s["error"]]

    use_color = color
    try:
        from colorama import Fore, Style
    except ImportError:
        use_color = False

    if use_color:
        from colorama import Fore, Style
        print(f"\n{Fore.MAGENTA}{'=' * 50}{Style.RESET_ALL}", file=sys.stderr)
        print(f"{Fore.MAGENTA}[trace] 请求追踪汇总{Style.RESET_ALL}", file=sys.stderr)
        for span in span_chain:
            icon = "X" if span["error"] else "+"
            c = Fore.RED if span["error"] else Fore.GREEN
            print(f"  {c}[{icon}] {span['node']} ({span['duration_ms']:.1f}ms){Style.RESET_ALL}", file=sys.stderr)
        print(f"{Fore.MAGENTA}[trace] 总计: {total_ms:.1f}ms, 错误数: {len(errors)}{Style.RESET_ALL}", file=sys.stderr)
        print(f"{Fore.MAGENTA}{'=' * 50}{Style.RESET_ALL}\n", file=sys.stderr)
    else:
        print(f"\n[trace] 请求追踪汇总:", file=sys.stderr)
        for span in span_chain:
            status = "ERROR" if span["error"] else "OK"
            print(f"  [{status}] {span['node']} ({span['duration_ms']:.1f}ms)", file=sys.stderr)
        print(f"[trace] 总计: {total_ms:.1f}ms, 错误数: {len(errors)}\n", file=sys.stderr)
