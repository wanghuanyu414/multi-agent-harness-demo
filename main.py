#!/usr/bin/env python3
"""旅行助手 Agent —— 交互式命令行入口"""
import asyncio
import logging
import os
import sys
import re
import uuid
import contextvars
from pathlib import Path

import yaml
from dotenv import load_dotenv
from colorama import init, Fore, Style
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from agent.mcp_client import MCPClient
from agent.builtin_tools import BUILTIN_TOOLS
from agent.graph import build_graph
from agent.memory import LongTermMemory
from agent.langsmith_client import init_langsmith
from agent.checkpoint import build_checkpointer
from agent.metrics import TokenUsageCallbackHandler
from agent.config_validate import validate_config, ConfigError

init(autoreset=True)
load_dotenv()

# Windows 控制台 UTF-8 编码支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# 请求关联 ID：web 层 middleware 写入，CLI 默认 "-"，贯穿日志便于追踪单次请求
request_id_var = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

# ============================================================
# 配置加载
# ============================================================

def _resolve_env(value: str) -> str:
    """解析 ${VAR:default} 格式的环境变量引用"""
    if not isinstance(value, str):
        return value

    def _replacer(match):
        var_expr = match[1]
        if ":" in var_expr:
            var_name, default = var_expr.split(":", 1)
            return os.environ.get(var_name.strip(), default.strip())
        return os.environ.get(var_expr, "")

    return re.sub(r"\$\{([^}]+)\}", _replacer, value)


def _resolve_config(obj):
    """递归解析配置对象中的环境变量引用"""
    if isinstance(obj, str):
        return _resolve_env(obj)
    if isinstance(obj, dict):
        return {k: _resolve_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_config(v) for v in obj]
    return obj


def load_config(path: str = "config.yaml") -> dict:
    """加载并解析 YAML 配置文件"""
    config_path = Path(__file__).parent / path
    if not config_path.exists():
        print(f"{Fore.RED}❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return _resolve_config(config)


# ============================================================
# 初始化
# ============================================================

def setup_logging(level: str = "INFO") -> None:
    """配置日志"""
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RequestIdFilter())
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[handler],
    )
    # 降低第三方库日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _build_guardrail_llm(config: dict, main_llm):
    """构建护栏专用 LLM；未配置 API key 时回退复用主 LLM。"""
    gc = config.get("guardrail_llm") or {}
    key = (gc.get("api_key") or "").strip()
    if not key or key.startswith("sk-your-") or key == "sk-...":
        logger.warning("护栏 LLM 未配置 API key，回退复用主 LLM")
        return main_llm

    provider = str(gc.get("provider", "openai")).lower()
    temperature = float(gc.get("temperature", 0.0))
    max_tokens = int(gc.get("max_tokens", 512))

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=gc.get("model", "deepseek-v4-flash"),
            anthropic_api_key=key,
            anthropic_api_url=gc.get("base_url", "https://api.deepseek.com/anthropic"),
            temperature=temperature,
            max_tokens=max_tokens,
            thinking={"type": "disabled"},
            callbacks=[TokenUsageCallbackHandler()],
        )
    return ChatOpenAI(
        model=gc.get("model", "deepseek-chat"),
        api_key=key,
        base_url=gc.get("base_url", "https://api.deepseek.com/v1"),
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=[TokenUsageCallbackHandler()],
    )


async def init_system(config: dict):
    """初始化 LLM 和 MCP 客户端"""
    # 配置校验
    validate_config(config)

    # 初始化 LangSmith
    init_langsmith(config.get("langsmith", {}))

    llm_config = config["llm"]

    # 检查 API Key
    api_key = llm_config.get("api_key", "")
    if not api_key or api_key == "sk-your-api-key-here":
        print(f"{Fore.YELLOW}⚠️  未设置 OPENAI_API_KEY，请在 .env 文件中配置")
        print(f"{Fore.YELLOW}   可以复制 .env.example 为 .env 并填入你的 API Key")
        sys.exit(1)

    llm = ChatOpenAI(
        model=llm_config.get("model", "gpt-4o"),
        api_key=api_key,
        base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 2048),
        callbacks=[TokenUsageCallbackHandler()],
    )

    # 护栏专用 LLM（未配置则回退主 LLM）
    guardrail_llm = _build_guardrail_llm(config, llm)

    # 初始化 MCP 客户端（传入内置工具）
    mcp_client = MCPClient(config.get("mcp_servers", []), builtin_tools=BUILTIN_TOOLS)
    await mcp_client.start_all()

    # 长期记忆
    memory = LongTermMemory(config.get("agent", {}).get("memory_path", "memory_store.json"))

    # 会话检查点
    checkpointer, close_checkpointer = await build_checkpointer(config)

    # 构建工作流（单 Agent 或多 Agent，取决于配置开关）
    if config.get("agent", {}).get("multi_agent", False):
        from agent.multiagent.graph import build_multi_agent_graph
        app = build_multi_agent_graph(llm, mcp_client, config, memory=memory,
                                      checkpointer=checkpointer, guardrail_llm=guardrail_llm)
    else:
        app = build_graph(llm, mcp_client, config, checkpointer=checkpointer, memory=memory,
                          guardrail_llm=guardrail_llm)

    return app, mcp_client, close_checkpointer


# ============================================================
# 交互式对话
# ============================================================

WELCOME = f"""
{Fore.CYAN}{'=' * 50}
{Fore.YELLOW}🚄✈️🌤️  旅行助手 Agent  {Fore.CYAN}
{Fore.WHITE}基于 LangGraph 1.0 + MCP 协议
{Fore.CYAN}{'=' * 50}

{Fore.GREEN}支持功能：
  {Fore.WHITE}🌤️  天气查询 —— 如「北京天气怎么样」
  {Fore.WHITE}🚄 火车票查询 —— 如「明天从上海到北京的高铁」
  {Fore.WHITE}💬 闲聊对话

{Fore.MAGENTA}输入 {Fore.YELLOW}help{Fore.MAGENTA} 查看更多用法，输入 {Fore.YELLOW}quit{Fore.MAGENTA} 退出
"""

HELP_TEXT = f"""
{Fore.CYAN}{'─' * 40}
{Fore.YELLOW}📖 使用帮助
{Fore.CYAN}{'─' * 40}

{Fore.GREEN}天气查询示例：
  {Fore.WHITE}• 北京天气怎么样
  {Fore.WHITE}• 查询上海的天气
  {Fore.WHITE}• 深圳今天热不热

{Fore.GREEN}车票查询示例：
  {Fore.WHITE}• 明天从北京到上海的高铁
  {Fore.WHITE}• 查询1月15日广州到深圳的火车票
  {Fore.WHITE}• 后天杭州到南京的票

{Fore.GREEN}命令：
  {Fore.WHITE}• {Fore.YELLOW}help{Fore.WHITE}  - 显示本帮助
  {Fore.WHITE}• {Fore.YELLOW}tools{Fore.WHITE} - 显示可用 MCP 工具
  {Fore.WHITE}• {Fore.YELLOW}status{Fore.WHITE} - 显示服务器运行状态
  {Fore.WHITE}• {Fore.YELLOW}quit{Fore.WHITE} - 退出程序
{Fore.CYAN}{'─' * 40}
"""


def _ask_human_review(payload: dict) -> dict:
    """人工审核交互：展示待审回复，返回 resume 决策值"""
    intent = payload.get("intent", "")
    response = payload.get("response", "")
    print(f"\n{Fore.YELLOW}⚠️  人工审核 [{intent}]{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{response}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}(y) 通过   (n) 拒绝   (e) 编辑后通过{Style.RESET_ALL}")
    ans = input(f"{Fore.GREEN}👤 审核: {Style.RESET_ALL}").strip().lower()
    if ans in ("y", "yes", ""):
        return {"approve": True}
    if ans in ("n", "no"):
        return {"approve": False}
    if ans in ("e", "edit"):
        edited = input(f"{Fore.GREEN}👤 修改后内容: {Style.RESET_ALL}").strip()
        return {"approve": True, "response": edited}
    # 其他输入一律视为拒绝
    return {"approve": False}


async def _resolve_human_review(app, result: dict, graph_config: dict) -> dict:
    """处理 Human-in-the-loop 中断：逐个取出 __interrupt__，人工决策后 resume"""
    while isinstance(result, dict) and result.get("__interrupt__"):
        interrupts = result["__interrupt__"]
        payload = interrupts[0].value if interrupts else {}
        if isinstance(payload, dict) and payload.get("type") == "human_review":
            decision = _ask_human_review(payload)
            result = await app.ainvoke(Command(resume=decision), graph_config)
        else:
            logger.warning(f"未知中断类型，跳过: {payload}")
            break
    return result


async def run_conversation(app, mcp_client: MCPClient, config: dict) -> None:
    """运行交互式对话循环"""
    print(WELCOME)

    # 会话配置（thread_id 用于多轮对话记忆；每次启动独立会话）
    thread_id = str(uuid.uuid4())
    graph_config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input(f"{Fore.GREEN}👤 你: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}👋 再见！祝旅途愉快～")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{Fore.YELLOW}👋 再见！祝旅途愉快～")
            break

        if user_input.lower() == "help":
            print(HELP_TEXT)
            continue

        if user_input.lower() == "tools":
            print(f"\n{Fore.CYAN}🔧 可用工具：")
            print(f"{Fore.WHITE}{mcp_client.get_all_tools_description()}\n")
            continue

        if user_input.lower() == "status":
            print(f"\n{Fore.CYAN}📡 服务器状态：")
            for name in ["weather", "12306"]:
                running = mcp_client.is_server_running(name)
                icon = f"{Fore.GREEN}✅" if running else f"{Fore.RED}❌"
                label = "运行中" if running else "未启动"
                print(f"  {icon} {name}: {label}")
            print()
            continue

        # 调用 Agent
        print(f"{Fore.CYAN}🤔 思考中...{Style.RESET_ALL}")
        try:
            result = await app.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                {**graph_config, "metadata": {"source": "cli"}},
            )

            # 处理人工审核中断（如有），resume 后得到最终结果
            result = await _resolve_human_review(app, result, graph_config)

            response = result.get("final_response", "😅 抱歉，出了点问题...")
            intent = result.get("intent", "chat")

            # 显示意图标签
            intent_label = {"weather": "🌤️ 天气", "ticket": "🚄 车票", "chat": "💬 对话"}.get(intent, "")
            if intent_label:
                print(f"{Fore.BLUE}[{intent_label}]{Style.RESET_ALL}")

            print(f"{Fore.YELLOW}🤖 助手: {Style.RESET_ALL}{response}\n")

            # 显示规划推理（CoT）
            plan_reasoning = result.get("plan_reasoning", "")
            if plan_reasoning:
                print(f"{Fore.CYAN}💭 推理: {plan_reasoning}{Style.RESET_ALL}")

            # 显示多 Agent 协作链路
            agent_trace = result.get("agent_trace", [])
            if agent_trace:
                print(f"{Fore.MAGENTA}[多Agent] {' → '.join(agent_trace)}{Style.RESET_ALL}")

            # 显示 Harness trace 汇总
            span_chain = result.get("span_chain", [])
            if span_chain:
                from agent.harness.observability import emit_span_chain_summary
                obs_config = config.get("harness", {}).get("observability", {})
                emit_span_chain_summary(span_chain, obs_config.get("color", True))

        except Exception as e:
            logger.error(f"Agent 调用失败: {e}", exc_info=True)
            print(f"{Fore.RED}❌ 出错了: {e}\n")


# ============================================================
# 主入口
# ============================================================

async def main() -> None:
    """主函数"""
    # 加载配置
    config = load_config()

    # 日志
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level)

    print(f"{Fore.CYAN}🚀 正在初始化系统...{Style.RESET_ALL}")

    app, mcp_client, close_checkpointer = None, None, None
    try:
        app, mcp_client, close_checkpointer = await init_system(config)
        await run_conversation(app, mcp_client, config)
    except ConfigError as e:
        logger.error(f"配置错误: {e}")
        print(f"{Fore.RED}❌ 配置错误:\n{e}")
    except Exception as e:
        logger.error(f"系统初始化失败: {e}", exc_info=True)
        print(f"{Fore.RED}❌ 系统初始化失败: {e}")
    finally:
        if mcp_client:
            print(f"{Fore.CYAN}🔌 正在关闭 MCP 连接...{Style.RESET_ALL}")
            await mcp_client.close_all()
        if close_checkpointer:
            await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())