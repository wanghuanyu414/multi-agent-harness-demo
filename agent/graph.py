"""LangGraph 工作流图构建 —— 支持 Harness 驾驭工程层"""
import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from .state import AgentState
from .mcp_client import MCPClient
from .nodes import (
    analyze_intent_node,
    execute_tool_node,
    generate_response_node,
)

logger = logging.getLogger(__name__)


def _make_analyze_intent(llm: ChatOpenAI, tools, tool_domain: dict):
    async def node(state: AgentState):
        return await analyze_intent_node(state, llm, tools, tool_domain)
    return node


def _make_execute_tool(mcp_client: MCPClient, tool_server: dict, use_format_script: bool = True):
    async def node(state: AgentState):
        return await execute_tool_node(state, mcp_client, tool_server, use_format_script=use_format_script)
    return node


def _make_generate_response(llm: ChatOpenAI, memory=None, review_intents=None):
    async def node(state: AgentState):
        return await generate_response_node(state, llm, memory=memory, review_intents=review_intents)
    return node


def build_graph(llm: ChatOpenAI, mcp_client: MCPClient,
                config: dict | None = None, checkpointer=None, memory=None,
                guardrail_llm=None) -> StateGraph:
    """构建 LangGraph 工作流图

    基础工作流:
        START → summarize → analyze_intent → route_tool → execute_tool → generate_response → END

    Harness 层（按配置启用）:
        summarize → analyze_intent → input_guardrail → [route_tool | plan_execution]
            → execute_tool → reflect ⇄ execute_tool → generate_response
            → output_guardrail → human_review → END

    Args:
        llm: ChatOpenAI 实例
        mcp_client: MCP 客户端
        config: 完整配置（含 harness / context / agent）；None 表示全部禁用
        memory: 长期记忆实例（可选，None 不启用）

    Returns:
        编译后的 StateGraph
    """
    guardrail_llm = guardrail_llm or llm  # 未传入则回退主 LLM
    hc = (config or {}).get("harness", {})
    summarize_config = (config or {}).get("context", {}).get("summarize", {})
    use_format_script = bool((config or {}).get("streaming", {}).get("ticket_format_script", False))
    human_review_config = hc.get("human_review", {})
    has_human_review = bool(human_review_config.get("enabled", False))
    review_intents = set(human_review_config.get("intents", [])) if has_human_review else set()

    # ── 解析各层开关 ──────────────────────────────────────
    g_config = hc.get("guardrails", {})
    has_input_guard = g_config.get("enabled", False) and g_config.get("input", {}).get("enabled", True)
    has_output_guard = g_config.get("enabled", False) and g_config.get("output", {}).get("enabled", True)
    has_planner = hc.get("planner", {}).get("enabled", False)
    has_reflection = hc.get("reflection", {}).get("enabled", False)
    obs_config = hc.get("observability", {})
    has_obs = obs_config.get("enabled", False)

    # ── 可观测性装饰器 ────────────────────────────────────
    def _trace(name, fn):
        if has_obs:
            from agent.harness.observability import trace_node
            fn = trace_node(name, obs_config)(fn)
        # LangSmith 节点追踪（与自研 trace_node 并存，互不影响）
        from agent.langsmith_client import is_enabled as _ls_enabled, traceable_node as _ls_trace
        if _ls_enabled():
            fn = _ls_trace(name)(fn)
        return fn

    # ── 规划节点工厂 ──────────────────────────────────────
    def _make_planner():
        from agent.harness.planner import planner_node
        planner_config = hc.get("planner", {})

        async def node(state: AgentState):
            return await planner_node(state, llm, mcp_client, planner_config)
        return node

    # ── 护栏节点工厂 ──────────────────────────────────────
    def _make_input_guardrail():
        from agent.harness.guardrails import input_guardrail_node
        guard_config = g_config.get("input", {})

        async def node(state: AgentState):
            result = await input_guardrail_node(state, guard_config, guardrail_llm)
            gr = result.get("input_guardrail_result", {})
            if gr.get("passed"):
                # 通过时清空 error
                result["error"] = ""
            return result
        return node

    def _make_output_guardrail():
        from agent.harness.guardrails import output_guardrail_node
        guard_config = g_config.get("output", {})

        async def node(state: AgentState):
            return await output_guardrail_node(state, guard_config, guardrail_llm)
        return node

    # ── 反思节点工厂 ──────────────────────────────────────
    def _make_reflect():
        from agent.harness.reflection import reflect_node
        reflect_config = hc.get("reflection", {})

        async def node(state: AgentState):
            return await reflect_node(state, reflect_config, llm)
        return node

    # ── 上下文压缩节点工厂（复用多 Agent 组件） ──────────────
    def _make_summarize(llm: ChatOpenAI):
        from agent.multiagent.supervisor import make_summarize_node
        return make_summarize_node(llm, summarize_config)

    # ── 人工审核节点工厂（复用多 Agent 组件） ────────────────
    def _make_human_review():
        from agent.multiagent.human_review import make_human_review_node
        return make_human_review_node(human_review_config)

    # ── 参数规范化子图（L0 确定性 + L2 强制工具，LLM 失败回退 L0） ──
    def _make_normalize():
        from agent.harness.param_normalize import build_param_normalize_graph
        return build_param_normalize_graph(llm)

    # ── 构建工具列表 + 工具名→服务器映射（function calling 用） ──
    from .multiagent.tools import build_specialist_tools
    specialist_tools = build_specialist_tools(mcp_client)
    all_tools = [t for domain_tools in specialist_tools.values() for t in domain_tools]
    # 意图识别只暴露入口工具
    entry_tools = {"get_weather", "query-tickets"}
    tools = [t for t in all_tools if t.name in entry_tools]
    # 工具名 → 意图领域，供 analyze_intent 做确定性映射（不靠字符串猜）
    tool_domain = {t.name: domain for domain, ts in specialist_tools.items() for t in ts}
    tool_server = {meta["name"]: meta["server"] for meta in mcp_client.list_tools()}

    # ── 构建图 ────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    # 注册核心节点
    workflow.add_node("summarize", _trace("summarize", _make_summarize(guardrail_llm)))
    workflow.add_node("analyze_intent", _trace("analyze_intent", _make_analyze_intent(guardrail_llm, tools, tool_domain)))
    workflow.add_node("execute_tool", _trace("execute_tool", _make_execute_tool(mcp_client, tool_server, use_format_script)))
    workflow.add_node("generate_response", _trace("generate_response", _make_generate_response(guardrail_llm, memory, review_intents)))

    # 注册 Harness 节点
    if has_input_guard:
        workflow.add_node("input_guardrail", _trace("input_guardrail", _make_input_guardrail()))

    if has_reflection:
        workflow.add_node("reflect", _trace("reflect", _make_reflect()))

    if has_output_guard:
        workflow.add_node("output_guardrail", _trace("output_guardrail", _make_output_guardrail()))

    if has_human_review:
        workflow.add_node("human_review", _trace("human_review", _make_human_review()))

    # ── 边的连接顺序（自上而下） ─────────────────────────

    # 1. START → summarize → analyze_intent
    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "analyze_intent")

    # 2. analyze_intent → [input_guardrail] → 分流（有工具调用 → execute_tool，否则 → generate_response）
    if has_input_guard:
        workflow.add_edge("analyze_intent", "input_guardrail")
        workflow.add_conditional_edges(
            "input_guardrail",
            _make_guard_router("execute_tool"),
            {"execute_tool": "execute_tool", "generate_response": "generate_response"},
        )
    else:
        workflow.add_conditional_edges(
            "analyze_intent",
            _make_tool_calls_router(),
            {"execute_tool": "execute_tool", "generate_response": "generate_response"},
        )

    # 3. execute_tool → 下一步（反思重试或直接生成回复）
    if has_reflection:
        workflow.add_edge("execute_tool", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            _make_reflect_router(hc.get("reflection", {})),
            {"execute_tool": "execute_tool", "generate_response": "generate_response"},
        )
    else:
        workflow.add_edge("execute_tool", "generate_response")

    # 6. generate_response → output_guardrail → human_review → END
    if has_output_guard:
        workflow.add_edge("generate_response", "output_guardrail")
        workflow.add_edge("output_guardrail", "human_review" if has_human_review else END)
    else:
        workflow.add_edge("generate_response", "human_review" if has_human_review else END)
    if has_human_review:
        workflow.add_edge("human_review", END)

    # ── 编译 ──────────────────────────────────────────────
    checkpointer = checkpointer or InMemorySaver()
    # 设置递归限制避免反射死循环: max_retries*2(execute+reflect) + 10(基础开销)
    recursion_limit = hc.get("reflection", {}).get("max_retries", 3) * 2 + 10
    app = workflow.compile(checkpointer=checkpointer)
    app = app.with_config({"recursion_limit": recursion_limit})
    logger.info(f"LangGraph 工作流编译完成 (harness: guard_in={has_input_guard}, "
                f"guard_out={has_output_guard}, planner={has_planner}, "
                f"reflect={has_reflection}, obs={has_obs}, summarize={summarize_config.get('enabled', True)}, "
                f"human_review={has_human_review}, memory={memory is not None})")
    return app


# ============================================================
# 条件路由函数（闭包捕获目标节点名）
# ============================================================

def _make_tool_calls_router():
    """normalize_params 之后：有工具调用 → execute_tool，否则 → generate_response"""
    def _route(state: AgentState) -> str:
        if state.get("tool_calls"):
            return "execute_tool"
        return "generate_response"
    return _route


def _make_guard_router(tool_target: str):
    """input_guardrail 之后：护栏拦截或无线工具调用 → generate_response，否则 → execute_tool"""
    def _route(state: AgentState) -> str:
        result = state.get("input_guardrail_result", {})
        if not result.get("passed") and result.get("mode") == "blocking":
            return "generate_response"
        if state.get("tool_calls"):
            return tool_target
        return "generate_response"
    return _route


def _make_reflect_router(config: dict):
    """reflect 之后的反思路由：retry → execute_tool，否则 → generate_response"""
    max_retries = config.get("max_retries", 3)

    def _route(state: AgentState) -> Literal["execute_tool", "generate_response"]:
        notes = state.get("reflection_notes", "")
        retry_count = state.get("retry_count", 0)
        if notes.startswith("retry:") and retry_count < max_retries:
            return "execute_tool"
        return "generate_response"
    return _route


def _step_router_node(state: AgentState) -> dict:
    """多步执行的纯路由节点：不修改状态，仅为条件边提供挂载点"""
    return {}


def _make_step_router():
    """多步执行路由：还有未执行的步骤 → execute_tool，否则 → generate_response"""
    def _route(state: AgentState) -> Literal["execute_tool", "generate_response"]:
        intent = state.get("intent", "chat")
        plan = state.get("execution_plan", [])
        current_step = state.get("current_step", 0)
        if intent in ("weather", "ticket") and plan and current_step < len(plan):
            return "execute_tool"
        return "generate_response"
    return _route
