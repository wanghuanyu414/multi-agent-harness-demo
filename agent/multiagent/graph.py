"""Supervisor + Worker 多 Agent 图构建"""
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_openai import ChatOpenAI

from .state import MultiAgentState
from .tools import build_specialist_tools
from .skills import SKILLS
from .supervisor import make_supervisor_node, make_agent_node, make_summarize_node, make_join_node
from .human_review import make_human_review_node

logger = logging.getLogger(__name__)


def _make_input_guardrail(config: dict, llm=None):
    """输入护栏节点：检测 PII/提示注入，blocking 模式直接拦截"""
    from agent.harness.guardrails import input_guardrail_node

    async def node(state) -> Command:
        result = await input_guardrail_node(state, config, llm)
        gr = result.get("input_guardrail_result", {"passed": True, "mode": config.get("mode", "warning")})
        if not gr.get("passed") and config.get("mode") == "blocking":
            return Command(
                update={
                    "input_guardrail_result": gr,
                    "final_response": result.get("error", "抱歉，您的输入包含不适合处理的内容，请修改后重试。"),
                },
                goto=END,
            )
        return Command(update={"input_guardrail_result": gr}, goto="summarize")
    return node


def _make_output_guardrail(config: dict, next_target, llm=None):
    """输出护栏节点：检测 PII 泄露/幻觉标记/事实性校验，blocking 模式替换回复"""
    from agent.harness.guardrails import output_guardrail_node

    async def node(state) -> Command:
        result = await output_guardrail_node(state, config, llm)
        gr = result.get("output_guardrail_result", {"passed": True, "mode": config.get("mode", "warning")})
        final = state.get("final_response", "")
        if not gr.get("passed") and config.get("mode") == "blocking":
            final = result.get("final_response", "抱歉，生成的回复包含不适合展示的内容。")
        return Command(update={"output_guardrail_result": gr, "final_response": final}, goto=next_target)
    return node


def build_multi_agent_graph(llm: ChatOpenAI, mcp_client, config: dict | None = None, memory=None, checkpointer=None,
                            guardrail_llm=None):
    """构建多 Agent 图：Supervisor（主管）+ 各专职 Worker + 护栏 + 人工审核

    Args:
        llm: 主 ChatOpenAI 实例（供各 Worker 使用）
        mcp_client: MCP 客户端（提供工具）
        config: 配置（含 harness.guardrails / human_review / recursion_limit 等）
        memory: 长期记忆实例（可选，None 表示不启用长期记忆）

    Returns:
        编译后的 StateGraph
    """
    config = config or {}
    guardrail_llm = guardrail_llm or llm  # 未传入则回退主 LLM
    harness = config.get("harness", {})

    # 人工审核（Human-in-the-loop）配置
    human_review_config = harness.get("human_review", {})
    human_review_enabled = bool(human_review_config.get("enabled", False))
    review_intents = set(human_review_config.get("intents", [])) if human_review_enabled else set()

    # 车票格式化脚本开关
    use_format_script = bool(config.get("streaming", {}).get("ticket_format_script", False))

    # 护栏配置
    guard_config = harness.get("guardrails", {})
    has_input_guard = bool(guard_config.get("enabled", False)) and bool(guard_config.get("input", {}).get("enabled", True))
    has_output_guard = bool(guard_config.get("enabled", False)) and bool(guard_config.get("output", {}).get("enabled", True))

    # 可观测性配置（节点级 span 追踪）
    obs_config = harness.get("observability", {})
    has_obs = bool(obs_config.get("enabled", False))

    # 各专职 Agent 的工具（按领域分区）
    specialist_tools = build_specialist_tools(mcp_client)

    # Supervisor 路由用独立模型
    try:
        supervisor_llm = guardrail_llm.model_copy(update={"temperature": 0.0, "max_tokens": 4096})
    except Exception:
        supervisor_llm = guardrail_llm

    # 聚合后的后处理链：output_guardrail → human_review → END
    finish_target = "output_guardrail" if has_output_guard else ("human_review" if human_review_enabled else END)
    guard_next = "human_review" if human_review_enabled else END

    workflow = StateGraph(MultiAgentState)

    # 节点追踪：harness trace_node（本地 span 耗时）+ LangSmith（云端 span）
    from agent.langsmith_client import is_enabled as _ls_enabled, traceable_node as _ls_trace

    def _ls_wrap(name, fn):
        if has_obs:
            from agent.harness.observability import trace_node
            fn = trace_node(name, obs_config)(fn)
        return _ls_trace(name)(fn) if _ls_enabled() else fn

    if has_input_guard:
        workflow.add_node("input_guardrail", _ls_wrap("input_guardrail", _make_input_guardrail(guard_config.get("input", {}), guardrail_llm)))
    # 上下文压缩节点（长会话治理）：消息超阈值时压缩旧消息为摘要，未超时透传
    summarize_config = config.get("context", {}).get("summarize", {})
    workflow.add_node("summarize", _ls_wrap("summarize", make_summarize_node(llm, summarize_config)))
    workflow.add_node("supervisor", _ls_wrap("supervisor", make_supervisor_node(supervisor_llm, SKILLS, memory=memory, finish_target=finish_target)))

    # 并行派发路由：supervisor 决策后 fan-out 到多个 Worker
    agent_to_intent = {s.name: s.intent for s in SKILLS}

    def route_to_workers(state):
        from langgraph.types import Send
        agents = state.get("agents_to_dispatch", [])
        return [
            Send(a, {**state, "next": a, "intent": agent_to_intent.get(a, "chat")})
            for a in agents
        ]

    # 从技能注册表循环构建专职 Agent 节点
    for skill in SKILLS:
        tools = skill.collect_tools(specialist_tools)
        workflow.add_node(skill.name, _ls_wrap(skill.name, make_agent_node(llm, tools, skill, memory=memory, memory_llm=llm, review_intents=review_intents, agent_to_intent=agent_to_intent, use_format_script=use_format_script)))

    # 聚合节点：并行 Worker 完成后汇聚结果（fan-in）
    agent_label = {s.name: s.label for s in SKILLS}
    workflow.add_node("join", _ls_wrap("join", make_join_node(agent_label, finish_target)))

    if has_output_guard:
        workflow.add_node("output_guardrail", _ls_wrap("output_guardrail", _make_output_guardrail(guard_config.get("output", {}), guard_next, guardrail_llm)))
    if human_review_enabled:
        workflow.add_node("human_review", _ls_wrap("human_review", make_human_review_node(human_review_config, agent_to_intent)))

    # 控制流：START → input_guardrail（或 summarize）→ summarize → supervisor，
    # 其余由 Command(goto=...) 决定
    if has_input_guard:
        workflow.add_edge(START, "input_guardrail")
    else:
        workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "supervisor")
    workflow.add_conditional_edges("supervisor", route_to_workers)

    checkpointer = checkpointer or InMemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    app = app.with_config({"recursion_limit": config.get("recursion_limit", 25)})
    logger.info(f"多 Agent 图编译完成 (Supervisor + {', '.join(s.name for s in SKILLS)})")
    return app
