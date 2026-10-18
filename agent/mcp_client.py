"""MCP 客户端封装 —— 支持多服务器、stdio 传输、JSON-RPC 协议"""
import asyncio
import json
import logging
import random
import time
from typing import Any

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from .tool_errors import ToolCallError, classify_exception, NOT_FOUND, CONNECTION

logger = logging.getLogger(__name__)


def _tool_to_dict(tool) -> dict:
    """官方 SDK 的 Tool 对象 → 现有 dict 格式（camelCase，兼容下游 tools.py）"""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "inputSchema": tool.input_schema or {},
    }


def _call_result_to_dict(result) -> dict:
    """官方 SDK 的 CallToolResult → dict，保持 {"content": [...], "isError": ...} 结构"""
    content = []
    for item in getattr(result, "content", None) or []:
        if getattr(item, "type", None) == "text" or hasattr(item, "text"):
            content.append({"type": "text", "text": getattr(item, "text", "")})
        else:
            # 图片 / 资源等非文本内容：尽量保留原始结构
            try:
                content.append(item.model_dump())
            except Exception:
                content.append({"type": getattr(item, "type", "unknown")})
    return {"content": content, "isError": bool(getattr(result, "isError", False))}


class MCPServer:
    """单个 MCP 服务器连接（基于官方 mcp SDK，统一 stdio / Streamable HTTP 传输）"""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.transport = str(config.get("transport", "stdio"))
        self.command = config.get("command")
        self.args = list(config.get("args", []) or [])
        self.url = config.get("url")
        self.timeout = float(config.get("timeout", 30.0))
        self._client: Client | None = None
        self._http_client: httpx2.AsyncClient | None = None
        self._tools: list[dict] = []
        self._alive = False
        self._listen_task: asyncio.Task | None = None
        self._on_tools_changed = None  # 工具列表变更回调（由 MCPClient 注入）

    def set_tools_changed_callback(self, callback) -> None:
        """注入工具列表变更回调，供热更新时回写 MCPClient 的工具注册表"""
        self._on_tools_changed = callback

    def _build_transport(self):
        if self.transport == "http":
            if not self.url:
                raise ValueError(f"[{self.name}] transport=http 缺少 url")
            # headers/timeout 必须放 httpx2.AsyncClient（streamable_http_client 不接受这些参数）
            self._http_client = httpx2.AsyncClient(
                timeout=httpx2.Timeout(self.timeout, read=self.timeout),
            )
            return streamable_http_client(self.url, http_client=self._http_client)
        if self.transport != "stdio":
            logger.warning(f"[{self.name}] 未知 transport={self.transport}，回退 stdio")
        if not self.command:
            raise ValueError(f"[{self.name}] transport=stdio 缺少 command")
        # 子进程默认 GBK 输出日志会乱码，强制 PYTHONIOENCODING=utf-8 让 stderr 用 UTF-8
        return stdio_client(StdioServerParameters(
            command=self.command,
            args=self.args,
            env={"PYTHONIOENCODING": "utf-8"},
        ))

    async def start(self) -> None:
        """建立连接、完成握手并拉取工具列表"""
        logger.info(f"[{self.name}] 正在启动 MCP 服务器 (transport={self.transport})")
        transport = self._build_transport()
        self._client = Client(transport, read_timeout_seconds=self.timeout)
        await self._client.__aenter__()
        result = await self._client.list_tools()
        self._tools = [_tool_to_dict(t) for t in result.tools]
        self._alive = True
        logger.info(
            f"[{self.name}] 启动成功，注册 {len(self._tools)} 个工具: "
            f"{[t['name'] for t in self._tools]}"
        )
        # 热更新订阅：服务器不支持 subscriptions/listen 时静默降级（靠 refresh_tools 兜底）
        self._listen_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self) -> None:
        """监听 tools/list_changed 通知，收到后刷新工具列表"""
        try:
            async with self._client.listen(tools_list_changed=True) as sub:
                async for _ in sub:
                    logger.info(f"[{self.name}] 收到 tools/list_changed，刷新工具列表")
                    await self.refresh_tools()
        except Exception as e:
            logger.debug(f"[{self.name}] list_changed 订阅不可用（降级惰性刷新）: {e}")

    async def refresh_tools(self) -> list[dict]:
        """重新拉取工具列表（热更新核心方法），并回写 MCPClient 工具注册表"""
        result = await self._client.list_tools()
        self._tools = [_tool_to_dict(t) for t in result.tools]
        if self._on_tools_changed:
            self._on_tools_changed(self.name, self._tools)
        return self._tools

    @property
    def tools(self) -> list[dict]:
        """已注册的工具列表"""
        return self._tools

    def is_alive(self) -> bool:
        """连接是否存活。进程中途退出导致的失活由 call_tool 异常 + 重试兜底"""
        return self._alive

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """调用指定工具，返回 {"content": [...], "isError": ...} 结构"""
        logger.info(f"[{self.name}] 调用工具 {tool_name}({arguments})")
        result = await self._client.call_tool(tool_name, arguments)
        content = _call_result_to_dict(result)
        logger.debug(f"[{self.name}] 工具结果: {json.dumps(content, ensure_ascii=False)[:300]}")
        return content

    async def close(self) -> None:
        """关闭连接"""
        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        if self._client:
            logger.info(f"[{self.name}] 正在关闭...")
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"[{self.name}] 关闭异常（忽略）: {e}")
            self._client = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._alive = False
        logger.info(f"[{self.name}] 已关闭")


class MCPClient:
    """多服务器 MCP 客户端管理器（同时支持 MCP 服务器和内置工具）"""

    def __init__(self, server_configs: list[dict],
                 builtin_tools: dict | None = None):
        """
        Args:
            server_configs: MCP 服务器配置列表，每项含 name, command, args
            builtin_tools: 内置工具字典 {tool_name: {name, description, inputSchema, handler}}
        """
        self._servers: dict[str, MCPServer] = {}
        self._configs = server_configs
        self._tool_registry: dict[str, tuple[str, dict]] = {}
        """工具名 -> (server_name, tool_schema)"""
        self._builtin_tools: dict[str, dict] = builtin_tools or {}
        """内置工具注册表"""
        self._builtin_handlers: dict[str, callable] = {}
        """工具名 -> 处理函数"""
        self._startup_errors: dict[str, str] = {}
        """服务器名 -> 启动失败原因"""
        self._server_domains: dict[str, str] = {}
        """服务器名 -> 领域归属（从 config 的 domain 字段读，供多 Agent 按领域分发工具）"""
        for cfg in server_configs:
            if not cfg.get("domain"):
                continue
            domain = str(cfg["domain"])
            if cfg.get("type") == "builtin":
                # 内置工具统一注册在 "builtin" server 名下，domain 也映射到 "builtin"
                self._server_domains["builtin"] = domain
            else:
                self._server_domains[str(cfg.get("name"))] = domain

    async def start_all(self, max_retries: int = 3,
                        retry_delay: float = 2.0) -> None:
        """启动所有配置的 MCP 服务器（单个失败不影响其他，支持重试）

        Args:
            max_retries: 单服务器最大重试次数（默认 3 次）
            retry_delay: 重试间隔基础秒数（指数退避：delay * 2^attempt）
        """
        for config in self._configs:
            if not config.get("enabled", True):
                logger.info(f"MCP 服务器 [{config['name']}] 已禁用，跳过")
                continue
            # 内置工具不需要启动
            if config.get("type") == "builtin":
                continue
            # 从配置读取重试参数（可覆盖默认值）
            cfg_retries = config.get("max_retries", max_retries)
            cfg_delay = config.get("retry_delay", retry_delay)

            server = MCPServer(config["name"], config)
            server.set_tools_changed_callback(self._on_server_tools_changed)
            started = False
            last_error = None

            for attempt in range(1, cfg_retries + 1):
                try:
                    await server.start()
                    self._servers[str(config["name"])] = server
                    for tool in server.tools:
                        self._tool_registry[tool["name"]] = (str(config["name"]), tool)
                    started = True
                    break
                except Exception as e:
                    last_error = e
                    if attempt < cfg_retries:
                        wait = cfg_delay * (2 ** (attempt - 1))  # 指数退避
                        logger.warning(
                            f"MCP 服务器 [{config['name']}] 启动失败 "
                            f"(第 {attempt}/{cfg_retries} 次), "
                            f"{wait:.1f}s 后重试: {e}"
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error(
                            f"MCP 服务器 [{config['name']}] 启动失败 "
                            f"(已重试 {cfg_retries} 次)，跳过: {e}"
                        )

            if not started:
                self._startup_errors[str(config["name"])] = str(last_error)
                if config.get("required", False):
                    raise RuntimeError(
                        f"必需的 MCP 服务器 [{config['name']}] 启动失败: {last_error}"
                    )
        # 注册内置工具
        for name, tool_def in self._builtin_tools.items():
            self._tool_registry[name] = ("builtin", {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "inputSchema": tool_def.get("inputSchema", {}),
            })
            self._builtin_handlers[name] = tool_def["handler"]
        logger.info(f"MCP 工具注册表: {list(self._tool_registry.keys())}")

    def _on_server_tools_changed(self, server_name: str, tools: list[dict]) -> None:
        """工具列表热更新回调：移除该 server 旧工具，注册新工具到 tool_registry"""
        for tool_name, (srv, _) in list(self._tool_registry.items()):
            if srv == server_name:
                del self._tool_registry[tool_name]
        for tool in tools:
            self._tool_registry[tool["name"]] = (server_name, tool)
        logger.info(
            f"MCP 服务器 [{server_name}] 工具列表热更新: {[t['name'] for t in tools]}"
        )

    def get_all_tools_description(self) -> str:
        """获取所有工具的描述文本（供 LLM 使用）"""
        lines = []
        for tool_name, (server_name, schema) in self._tool_registry.items():
            desc = schema.get("description", "无描述")
            params = schema.get("inputSchema", {}).get("properties", {})
            param_desc = ", ".join(
                f"{k}({v.get('type','str')})" for k, v in params.items()
            )
            lines.append(f"  - {tool_name} [{server_name}]: {desc}，参数: {param_desc}")
        return "\n".join(lines) if lines else "暂无可用工具"

    def list_tools(self) -> list[dict]:
        """返回所有已注册工具（含内置工具）的元信息，供多 Agent 构建工具使用

        Returns:
            [{"name": str, "server": str, "description": str, "inputSchema": dict}]
        """
        out: list[dict] = []
        for tool_name, (server_name, schema) in self._tool_registry.items():
            out.append({
                "name": tool_name,
                "server": server_name,
                "domain": self._server_domains.get(server_name, ""),
                "description": schema.get("description", ""),
                "inputSchema": schema.get("inputSchema", {}),
            })
        return out

    def is_server_running(self, server_name: str) -> bool:
        """检查指定服务器是否可用（进程存活）"""
        server_name = str(server_name)
        # 内置工具始终可用，无需服务进程
        if server_name == "builtin":
            return True
        # 先检查是否在配置中但类型为 builtin
        for config in self._configs:
            if str(config.get("name")) == server_name:
                if config.get("type") == "builtin":
                    return True
                # 非内置工具，检查实际进程存活
                for k, s in self._servers.items():
                    if str(k) == server_name:
                        return s.is_alive()
                return False
        # 不在配置中，也检查 _servers（兼容动态注册）
        for k, s in self._servers.items():
            if str(k) == server_name:
                return s.is_alive()
        return False

    def get_startup_error(self, server_name: str) -> str | None:
        """获取服务器启动失败的原因，如果正常运行则返回 None"""
        return self._startup_errors.get(str(server_name))

    def unhealthy_servers(self) -> list[str]:
        """返回进程已退出的 MCP 服务器名列表（就绪探针用）"""
        return [str(name) for name, s in self._servers.items() if not s.is_alive()]

    def find_tool(self, server_name: str, tool_hint: str = "") -> dict | None:
        """在指定服务器或内置工具中查找工具

        Args:
            server_name: 服务器名称（对内置工具也适用）
            tool_hint: 工具名称提示（模糊匹配）

        Returns:
            匹配的工具 schema，未找到返回 None
        """
        # 先查工具注册表
        for tool_name, (srv, schema) in self._tool_registry.items():
            if srv == server_name or srv == "builtin":
                if tool_hint:
                    if tool_hint.lower() in tool_name.lower():
                        return schema
                else:
                    return schema
        # 回退到 MCP 服务器
        server = self._servers.get(server_name)
        if server:
            tools = server.tools
            if tools:
                if tool_hint:
                    for t in tools:
                        if tool_hint.lower() in t["name"].lower():
                            return t
                return tools[0]
        return None

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict,
                        max_retries: int = 2) -> dict:
        """调用指定服务器上的工具（支持 MCP 和内置工具）。

        统一入口：无论内置工具还是 MCP 工具，都在此记录成功率/耗时/token 遥测。
        失败先分类（classify_exception）：只有可重试的（超时/连接/服务端错误）
        才做指数退避 + 抖动重试；确定性失败（参数错误/资源不存在）直接抛，不浪费重试。
        重试耗尽或不可重试时，抛 ToolCallError（携带错误码与用户友好文案）。
        """
        from .metrics import MetricsCollector, count_tokens
        metrics = MetricsCollector.get()

        for attempt in range(max_retries + 1):
            start = time.monotonic()
            try:
                # 内置工具
                if tool_name in self._builtin_handlers:
                    handler = self._builtin_handlers[tool_name]
                    result = await asyncio.to_thread(handler, **arguments)
                else:
                    # MCP 工具
                    server = self._servers.get(server_name)
                    if not server:
                        raise ToolCallError(
                            NOT_FOUND, "服务未注册，请稍后重试", retryable=False,
                            detail=f"MCP 服务器 '{server_name}' 未找到。可用: {list(self._servers.keys())}",
                        )
                    if not server.is_alive():
                        raise ToolCallError(
                            CONNECTION, "服务暂时不可用，请稍后重试", retryable=True,
                            detail=f"MCP 服务器 '{server_name}' 已断开",
                        )
                    result = await server.call_tool(tool_name, arguments)
            except ToolCallError as e:
                err = e
            except Exception as e:
                err = classify_exception(e)
            else:
                metrics.record_tool(tool_name, True, (time.monotonic() - start) * 1000, count_tokens(str(result)))
                return result

            metrics.record_tool(tool_name, False, (time.monotonic() - start) * 1000)

            if not err.retryable:
                logger.warning("工具 %s 调用失败(不可重试) [%s]: %s", tool_name, err.code, err.detail)
                raise err

            if attempt < max_retries:
                # 指数退避 + 全抖动
                wait = (0.5 * (2 ** attempt)) * (0.5 + random.random())
                logger.warning(
                    "工具 %s 调用失败(第 %d/%d 次) [%s]，%.2fs 后重试: %s",
                    tool_name, attempt + 1, max_retries + 1, err.code, wait, err.detail,
                )
                await asyncio.sleep(wait)
                continue
            raise err

    async def close_all(self) -> None:
        """关闭所有 MCP 服务器连接"""
        for server in self._servers.values():
            await server.close()
        self._servers.clear()
        self._tool_registry.clear()