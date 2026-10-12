"""MCP 工具 → LangChain StructuredTool 转换，并按领域分发给专职 Agent"""
import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model

from ..tool_errors import ToolCallError

logger = logging.getLogger(__name__)

_JSON_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_type_to_python(type_name: Any) -> Any:
    """JSON Schema type → Python 类型（type 可能是数组，取第一个非 null）"""
    if isinstance(type_name, list):
        non_null = [t for t in type_name if t != "null"]
        type_name = non_null[0] if non_null else "string"
    return _JSON_TYPE_MAP.get(type_name, str)


def _build_args_model(tool_name: str, input_schema: dict) -> type[BaseModel]:
    """根据 MCP inputSchema 动态生成 Pydantic args model"""
    fields: dict[str, Any] = {}
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    for name, spec in properties.items():
        py_type = _json_type_to_python(spec.get("type", "string"))
        if name in required:
            fields[name] = (py_type, ...)
        else:
            fields[name] = (py_type, None)

    safe_name = "".join(c if c.isalnum() else "_" for c in tool_name)
    return create_model(f"{safe_name}_args", **fields)


def mcp_tool_to_langchain_tool(mcp_client, name: str, server: str, schema: dict) -> StructuredTool:
    """把单个 MCP 工具封装成 LangChain StructuredTool"""
    args_model = _build_args_model(name, schema.get("inputSchema", {}))

    async def _run(**kwargs):
        try:
            result = await mcp_client.call_tool(server, name, kwargs)
        except ToolCallError as e:
            # 返回用户友好文案而非抛异常，让 worker 的 ReAct 稳定拿到可读信息去降级
            logger.warning("工具 %s 调用失败 [%s]: %s", name, e.code, e.detail)
            return e.user_message
        except Exception as e:
            logger.error("工具 %s 调用异常: %s", name, e, exc_info=True)
            return "工具调用失败，请稍后重试"
        # 间接注入防护：替换为安全标记
        from ..harness.guardrails import sanitize_tool_output
        result, _ = sanitize_tool_output(result)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=schema.get("description", "") or f"调用 {server}.{name}",
        args_schema=args_model,
    )


# 服务器名 → 领域 分区映射（当前项目各只有 1 个工具，映射无歧义；
# 如需扩展，可改为从 config.yaml 的 mcp_servers 读取 domain 字段）
_SERVER_TO_DOMAIN = {
    "builtin": "weather",   # 内置天气工具注册在 "builtin" 名下
    "weather": "weather",
    "12306": "ticket",
}


def build_specialist_tools(mcp_client) -> dict[str, list[StructuredTool]]:
    """枚举 MCP 客户端工具，按领域返回 {domain: [tools]}"""
    result: dict[str, list] = {"weather": [], "ticket": []}
    for meta in mcp_client.list_tools():
        server = meta["server"]
        domain = _SERVER_TO_DOMAIN.get(server)
        if domain is None:
            n = meta["name"].lower()
            domain = "ticket" if any(k in n for k in ("ticket", "12306", "train")) else "weather"
        tool = mcp_tool_to_langchain_tool(mcp_client, meta["name"], server, meta)
        result[domain].append(tool)
        logger.info(f"工具 {meta['name']} [{server}] → {domain}_agent")
    return result
