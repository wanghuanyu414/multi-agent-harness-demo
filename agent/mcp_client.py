"""MCP 客户端封装 —— 支持多服务器、stdio 传输、JSON-RPC 协议"""
import asyncio
import json
import logging
import random
import time
from typing import Any

from .tool_errors import ToolCallError, classify_exception, NOT_FOUND, CONNECTION

logger = logging.getLogger(__name__)


class MCPServer:
    """单个 MCP 服务器连接

    通过 stdio 与 MCP 服务器进程通信，使用 JSON-RPC 2.0 协议。
    """

    def __init__(self, name: str, command: str, args: list[str], timeout: float = 30.0):
        """
        Args:
            name: 服务器名称（用于日志和标识）
            command: 启动命令（如 npx / 可执行文件绝对路径）
            args: 命令参数
            timeout: 单次工具调用的响应超时（秒）
        """
        self.name = name
        self.command = command
        self.args = args
        self.timeout = timeout
        self.process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._tools: list[dict] = []
        self._lock = asyncio.Lock()

    def is_alive(self) -> bool:
        """进程是否存活（未退出）"""
        return self.process is not None and self.process.returncode is None

    async def start(self) -> None:
        """启动 MCP 服务器进程并完成初始化握手"""
        logger.info(f"[{self.name}] 正在启动 MCP 服务器: {self.command} {' '.join(self.args)}")
        try:
            # 用 exec 启动子进程，command 为可执行文件绝对路径
            self.process = await asyncio.create_subprocess_exec(
                self.command, *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # 给进程一点时间启动，检查是否立即退出
            await asyncio.sleep(0.5)
            if self.process.returncode is not None:
                stderr_data = await self.process.stderr.read() if self.process.stderr else b""
                raise ConnectionError(
                    f"[{self.name}] 进程立即退出 (code={self.process.returncode})。"
                    f"stderr: {stderr_data.decode(errors='replace')[:500]}"
                )
            await self._initialize()
            self._tools = await self._list_tools()
            logger.info(
                f"[{self.name}] 启动成功，注册 {len(self._tools)} 个工具: "
                f"{[t.get('name') for t in self._tools]}"
            )
        except Exception as e:
            logger.error(f"[{self.name}] 启动失败: {e}")
            await self.close()
            raise

    async def _initialize(self) -> dict:
        """MCP 初始化握手"""
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "travel-agent", "version": "1.0.0"},
        })
        # 发送 initialized 通知
        await self._send_notification("notifications/initialized", {})
        return result

    async def _send_request(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求并等待响应"""
        async with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }
            return await self._send_and_receive(request)

    async def _send_notification(self, method: str, params: dict) -> None:
        """发送 JSON-RPC 通知（无响应）"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        async with self._lock:
            await self._write(notification)

    async def _send_and_receive(self, message: dict) -> dict:
        """写入消息并读取响应"""
        await self._write(message)
        return await self._read_response()

    async def _write(self, message: dict) -> None:
        """写入 JSON 消息到进程 stdin"""
        data = json.dumps(message, ensure_ascii=False) + "\n"
        logger.debug(f"[{self.name}] → {data.strip()}")
        assert self.process and self.process.stdin
        self.process.stdin.write(data.encode())
        await self.process.stdin.drain()

    async def _read_response(self) -> dict:
        """从进程 stdout 读取 JSON 响应（带超时；超时/断连抛明确异常）"""
        assert self.process and self.process.stdout
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"[{self.name}] MCP 响应超时（{self.timeout}s）")
        if not line:
            # 进程可能已退出，检查 stderr
            if self.process.stderr:
                stderr_data = await self.process.stderr.read()
                raise ConnectionError(
                    f"[{self.name}] MCP 服务器已断开。stderr: {stderr_data.decode(errors='replace')}"
                )
            raise ConnectionError(f"[{self.name}] MCP 服务器已断开，无错误输出")
        data = json.loads(line)
        logger.debug(f"[{self.name}] ← {json.dumps(data, ensure_ascii=False)[:500]}")
        if "error" in data:
            raise RuntimeError(f"[{self.name}] JSON-RPC 错误: {data['error']}")
        return data

    async def _list_tools(self) -> list[dict]:
        """获取服务器提供的工具列表"""
        result = await self._send_request("tools/list", {})
        return result.get("result", {}).get("tools", [])

    @property
    def tools(self) -> list[dict]:
        """已注册的工具列表"""
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """调用指定工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        logger.info(f"[{self.name}] 调用工具 {tool_name}({arguments})")
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        content = result.get("result", {})
        logger.debug(f"[{self.name}] 工具结果: {json.dumps(content, ensure_ascii=False)[:300]}")
        return content

    async def close(self) -> None:
        """关闭 MCP 服务器连接"""
        if self.process and self.process.returncode is None:
            logger.info(f"[{self.name}] 正在关闭...")
            try:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            except ProcessLookupError:
                pass  # 进程已退出
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

            server = MCPServer(config["name"], config["command"], config["args"],
                               timeout=float(config.get("timeout", 30.0)))
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