"""Supervisor 主管 Agent —— 多意图识别、多步路由调度、结果聚合"""
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.graph.message import RemoveMessage
from langgraph.types import Command
from pydantic import BaseModel

from ..memory import extract_facts
from ..prompts import SUPERVISOR_PROMPT_TEMPLATE, SUMMARIZE_PROMPT
from ..langsmith_client import get_prompt
from ..metrics import count_message_tokens
from .skills import Skill

logger = logging.getLogger(__name__)


def _make_route_model(members: list[str]) -> type[BaseModel]:
    """按技能名动态生成 Route 模型，用 Literal 枚举约束 agents 只能输出合法技能名"""
    AgentName = Literal[tuple(members)]

    class Route(BaseModel):
        """Supervisor 的多步路由决策：需要依次派发的 Worker 列表 + 派发推理（CoT）"""
        agents: list[AgentName]
        reasoning: str

    return Route


def _build_supervisor_prompt(skills: list[Skill]) -> str:
    """根据技能注册表动态生成 Supervisor 路由 prompt（模板从 Hub 拉取，失败回退本地）"""
    skills_desc = "\n".join(f"- {s.name}：{s.description}" for s in skills)
    return get_prompt("supervisor", SUPERVISOR_PROMPT_TEMPLATE).format(skills_description=skills_desc)


def _latest_user_text(messages) -> str:
    """从消息历史中取最新一条真实用户消息（不含 Worker 回写的带 name 消息）"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and not getattr(m, "name", None):
            return m.content if hasattr(m, "content") else str(m)
    return ""


def _messages_to_text(messages) -> str:
    """把消息序列渲染成可读文本（供压缩摘要 LLM 使用）"""
    lines = []
    for m in messages:
        name = getattr(m, "name", None)
        role = name or ("用户" if isinstance(m, HumanMessage) else "助手")
        content = m.content if hasattr(m, "content") else str(m)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def make_summarize_node(llm, config: dict | None = None):
    """构造上下文压缩节点：消息 token 超过预算时，把旧消息压成摘要并从状态中删除。

    触发条件：count_message_tokens(messages) > max_tokens。
    保留最近 keep_recent 条原文，其余压缩进 conversation_summary（与已有摘要合并）。
    压缩失败时静默跳过，不影响主流程。
    """
    config = config or {}
    enabled = bool(config.get("enabled", True))
    max_tokens = int(config.get("max_tokens", 6000))  # 按 token 预算触发压缩（更贴合实际窗口）
    keep_recent = max(int(config.get("keep_recent", 4)), 0)
    summarize_prompt = get_prompt("summarize", SUMMARIZE_PROMPT)

    async def summarize_node(state) -> dict:
        if not enabled:
            return {}
        messages = list(state.get("messages", []))
        if keep_recent == 0 or count_message_tokens(messages) <= max_tokens:
            return {}

        to_summarize = messages[:-keep_recent]
        if not to_summarize:
            return {}

        old_text = _messages_to_text(to_summarize)
        prev_summary = state.get("conversation_summary", "") or "（无）"
        prompt = summarize_prompt.format(prev_summary=prev_summary, conversation=old_text)

        try:
            r = await llm.ainvoke([
                SystemMessage(content="你是对话压缩模块，负责把长对话压成要点摘要。"),
                HumanMessage(content=prompt),
            ])
            new_summary = (r.content or "").strip()
        except Exception as e:
            logger.warning(f"上下文压缩失败，跳过本轮: {e}")
            return {}

        removed = [RemoveMessage(id=m.id) for m in to_summarize if getattr(m, "id", None)]
        logger.info(f"上下文压缩: 压缩 {len(to_summarize)} 条 → 保留 {keep_recent} 条")
        return {"conversation_summary": new_summary, "messages": removed}

    return summarize_node


def _aggregate_results(state, label_map: dict) -> str:
    """聚合多个 Worker 的回复为最终回复（带标题分段）"""
    results = state.get("worker_results", [])
    if not results:
        return state.get("final_response", "")
    if len(results) == 1:
        return results[0].get("response", "")
    parts = []
    for r in results:
        label = label_map.get(r.get("agent"), r.get("agent"))
        parts.append(f"### {label}\n{r.get('response', '')}")
    return "\n\n".join(parts)


def make_supervisor_node(llm, skills: list[Skill], memory=None, finish_target=END):
    """构造 Supervisor 节点：召回记忆 + 多意图识别 + 多步路由 + 结果聚合"""
    members = [s.name for s in skills]
    agent_to_intent = {s.name: s.intent for s in skills}
    agent_label = {s.name: s.label for s in skills}
    fallback = next((s.name for s in skills if s.intent == "chat"), members[0] if members else "chat_agent")
    supervisor_prompt = _build_supervisor_prompt(skills)
    Route = _make_route_model(members)
    structured_llm = llm.with_structured_output(Route)

    def supervisor(state) -> Command:
        messages = list(state.get("messages", []))
        last_is_worker = bool(messages) and bool(getattr(messages[-1], "name", None))

        # ── 情况 1：某个 Worker 刚完成 ──
        if last_is_worker:
            pending = list(state.get("pending_agents", []))
            if pending:
                # 还有待派发的 Worker，继续派发
                next_agent = pending.pop(0)
                trace = list(state.get("agent_trace", [])) + ["supervisor"]
                logger.info(f"Supervisor: 继续派发 → {next_agent}")
                return Command(
                    update={"next": next_agent, "pending_agents": pending, "agent_trace": trace},
                    goto=next_agent,
                )
            # 所有 Worker 完成 → 聚合结果并进入后处理链（output_guardrail / human_review / END）
            final = _aggregate_results(state, agent_label)
            logger.info("Supervisor: 所有 Worker 完成 → FINISH")
            return Command(update={"next": "FINISH", "final_response": final}, goto=finish_target)

        # ── 情况 2：首次路由（新的用户请求） ──
        user_text = _latest_user_text(messages)

        # 召回与当前用户消息相关的长期记忆
        memory_context = ""
        if memory:
            memories = memory.recall(user_text, top_k=5)
            memory_context = memory.to_context(memories)
            if memory_context:
                logger.info(f"Supervisor: 召回 {len(memories)} 条长期记忆")

        # 多意图识别：返回需要依次派发的 Agent 列表
        sys_msgs = [SystemMessage(content=supervisor_prompt)]
        summary = state.get("conversation_summary", "")
        if summary:
            sys_msgs.append(SystemMessage(content="## 对话历史摘要\n" + summary))
        route = structured_llm.invoke([*sys_msgs, *messages])
        agents = route.agents or [fallback]
        agents = [a for a in agents if a in members] or [fallback]
        first, pending = agents[0], agents[1:]
        intent = agent_to_intent.get(first, "chat")
        reasoning = (route.reasoning or "").strip()
        logger.info(f"Supervisor 决策: agents={agents}，推理: {reasoning}")

        return Command(
            update={
                "next": first,
                "intent": intent,
                "pending_agents": pending,
                "worker_results": [],  # 新的一轮，重置结果累积
                "memory_context": memory_context,
                "plan_reasoning": reasoning,
                "agent_trace": ["supervisor"],
            },
            goto=first,
        )
    return supervisor


def make_agent_node(agent, name: str, memory=None, memory_llm=None):
    """把编译后的专职 Agent 包装成 Supervisor 图里的节点

    Worker 内部 ReAct 轨迹留在自己的子图里，只把最终消息回写共享 messages；
    同时注入长期记忆上下文，并在完成后固化事实 + 累积结果。
    """
    async def agent_node(state) -> Command:
        # 注入长期记忆上下文（只影响本轮，不写入共享 messages）
        memory_context = state.get("memory_context", "")
        input_messages = list(state["messages"])
        prefix = []
        summary = state.get("conversation_summary", "")
        if summary:
            prefix.append(SystemMessage(content="## 对话历史摘要\n" + summary))
        if memory_context:
            prefix.append(SystemMessage(content="## 用户长期记忆\n" + memory_context))
        if prefix:
            input_messages = prefix + input_messages

        result = await agent.ainvoke({"messages": input_messages})
        last = result["messages"][-1]
        content = last.content if hasattr(last, "content") else str(last)
        logger.info(f"[{name}] 完成，回复长度 {len(content)}")

        # 提取 Worker 实际调用的工具原始结果，供输出护栏事实性校验
        tool_texts = []
        for m in result.get("messages", []):
            if isinstance(m, ToolMessage):
                c = m.content if hasattr(m, "content") else str(m)
                tool_texts.append(c if isinstance(c, str) else str(c))

        # 记忆固化：提取本轮事实写入长期记忆（失败不影响主流程）
        if memory and memory_llm:
            user_text = _latest_user_text(state.get("messages", []))
            try:
                facts = await extract_facts(memory_llm, user_text, content)
                for f in facts:
                    memory.remember(f, kind="fact")
            except Exception as e:
                logger.warning(f"[{name}] 记忆固化失败: {e}")

        # 累积 Worker 结果 + 工具结果 + trace
        worker_results = list(state.get("worker_results", []))
        worker_results.append({"agent": name, "response": content})
        tool_results = list(state.get("tool_results", [])) + tool_texts
        trace = list(state.get("agent_trace", [])) + [name]

        return Command(
            update={
                "messages": [HumanMessage(content=content, name=name)],
                "final_response": content,
                "worker_results": worker_results,
                "tool_results": tool_results,
                "agent_trace": trace,
            },
            goto="supervisor",
        )
    return agent_node
