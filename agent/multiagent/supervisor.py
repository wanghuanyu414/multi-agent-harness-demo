"""Supervisor 主管 Agent —— 多意图识别、多步路由调度、结果聚合"""
import asyncio
import json
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
from ..metrics import count_message_tokens, get_stream_queue
from .skills import Skill
from .skill_loader import run_skill_script

logger = logging.getLogger(__name__)


async def _run_skill_script(skill, script_name: str, input_text: str) -> str | None:
    """执行 skill 目录 scripts/ 下的脚本（第 3 级按需执行），薄包装 run_skill_script。"""
    md_path = getattr(skill, "_skill_md_path", None)
    if not md_path:
        return None
    return await run_skill_script(md_path, script_name, input_text)

# 多意图兜底关键词：路由未命中时，按领域关键词补发缺失的 Agent
_INTENT_BACKFILL_KEYWORDS = {
    "weather_agent": (
        "天气", "气温", "温度", "下雨", "晴天", "阴天", "雾霾", "空气质量",
        "湿度", "风力", "紫外线", "会不会下雨", "带伞",
    ),
    "ticket_agent": (
        "车票", "火车票", "高铁", "动车", "订票", "购票", "车次", "列车",
        "二等座", "一等座", "商务座", "硬座", "硬卧", "软卧", "余票", "12306",
        "火车", "铁路",
    ),
}


def _backfill_multi_intent(agents: list[str], user_text: str) -> list[str]:
    """多意图兜底：supervisor 做结构化路由，偶发对并列句式
    （如「帮我查天气 + 帮我查车票」）只识别出第一个意图。这里用关键词补发，
    保证天气/车票意图不因未命中而丢失。"""
    result = list(agents)
    for agent, keywords in _INTENT_BACKFILL_KEYWORDS.items():
        if agent not in result and any(kw in user_text for kw in keywords):
            result.append(agent)
    return result


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


def _aggregate_results(results, label_map: dict) -> str:
    """聚合多个 Worker 的回复为最终回复（带标题分段）"""
    if not results:
        return ""
    if len(results) == 1:
        return results[0].get("response", "")
    parts = []
    for r in results:
        label = label_map.get(r.get("agent"), r.get("agent"))
        parts.append(f"### {label}\n{r.get('response', '')}")
    return "\n\n".join(parts)


def make_join_node(agent_label: dict, finish_target=END):
    """聚合节点：所有 Worker 完成后聚合结果，进入后处理链。"""
    def join(state) -> Command:
        # 按 Supervisor 决策顺序排序 Worker 结果（并行完成顺序不定）
        order = {a: i for i, a in enumerate(state.get("agents_to_dispatch", []))}
        results = sorted(
            state.get("worker_results", []),
            key=lambda r: order.get(r.get("agent"), 999),
        )
        final = _aggregate_results(results, agent_label)
        logger.info("所有 Worker 完成 → 聚合")
        return Command(update={"next": "FINISH", "final_response": final}, goto=finish_target)
    return join


def make_supervisor_node(llm, skills: list[Skill], memory=None, finish_target=END):
    """构造 Supervisor 节点：召回记忆 + 多意图识别 + 多步路由 + 结果聚合"""
    members = [s.name for s in skills]
    agent_to_intent = {s.name: s.intent for s in skills}
    agent_label = {s.name: s.label for s in skills}
    fallback = next((s.name for s in skills if s.intent == "chat"), members[0] if members else "chat_agent")
    supervisor_prompt = _build_supervisor_prompt(skills)
    Route = _make_route_model(members)
    structured_llm = llm.with_structured_output(Route, method="function_calling")

    def supervisor(state) -> dict:
        messages = list(state.get("messages", []))

        # 首次路由（新的用户请求）
        user_text = _latest_user_text(messages)

        # 召回与当前用户消息相关的长期记忆
        memory_context = ""
        if memory:
            memories = memory.recall(user_text, top_k=5)
            memory_context = memory.to_context(memories)
            if memory_context:
                logger.info(f"Supervisor: 召回 {len(memories)} 条长期记忆")

        # 多意图识别：返回需要派发的 Agent 列表
        sys_msgs = [SystemMessage(content=supervisor_prompt)]
        summary = state.get("conversation_summary", "")
        if summary:
            sys_msgs.append(SystemMessage(content="## 对话历史摘要\n" + summary))
        route = structured_llm.invoke([*sys_msgs, *messages])
        agents = route.agents or [fallback]
        agents = [a for a in agents if a in members] or [fallback]
        agents = _backfill_multi_intent(agents, user_text)
        reasoning = (route.reasoning or "").strip()
        logger.info(f"Supervisor 决策: agents={agents}，推理: {reasoning}")

        # 返回待派发的 agents 列表，由 conditional_edges 做 fan-out 并行派发
        trace = ["supervisor", *agents]
        return {
            "agents_to_dispatch": agents,
            "intent": agent_to_intent.get(agents[0], "chat"),
            "memory_context": memory_context,
            "plan_reasoning": reasoning,
            "agent_trace": trace,
            "worker_results": [],
        }
    return supervisor


async def _async_remember(memory, memory_llm, user_text, content, name):
    """后台记忆固化：提取事实写入长期记忆"""
    try:
        facts = await extract_facts(memory_llm, user_text, content)
        for f in facts:
            memory.remember(f, kind="fact")
    except Exception as e:
        logger.warning(f"[{name}] 记忆固化失败: {e}")


async def _run_react(llm, tools, skill, prefix, input_messages, name, review_intents=None, turn_needs_review=None,
                     use_format_script=True):
    """自定义 ReAct：bind_tools 路由 → 执行工具 → 流式生成最终回复。

    turn_needs_review 表示「本轮是否需要人工审核」（整轮级判断）：传入时直接决定是否
    禁用流式；为 None 时回退到单 Agent 判断（skill.intent in review_intents）。
    """
    system_prompt = get_prompt(skill.intent, skill.prompt)
    messages = [SystemMessage(content=system_prompt)] + prefix + input_messages
    tools_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    tool_texts = []
    # 路由循环（非流式，最多 3 轮）
    for _ in range(3):
        response = await llm_with_tools.ainvoke(messages)
        if not getattr(response, "tool_calls", None):
            break
        messages.append(response)
        for tc in response.tool_calls:
            tool = tools_map.get(tc.get("name"))
            args = tc.get("args", {})
            if tool is None:
                result = f"工具 {tc.get('name')} 未找到"
            else:
                try:
                    r = await tool.ainvoke(args)
                    result = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
                except Exception as e:
                    result = f"工具 {tc.get('name')} 调用失败: {e}"
            # 车票查询结果用 skill 脚本格式化（第 3 级）
            # use_format_script=False 时跳过脚本，原始 JSON 交给 LLM 流式格式化
            if use_format_script and skill.intent == "ticket" and tc.get("name") == "query-tickets":
                # result 是 MCP wrapper（{"content":[{"text":"..."}]}），需先取出内层 query-tickets 原始 JSON
                fmt_input = result
                try:
                    parsed = json.loads(result) if isinstance(result, str) else result
                    if isinstance(parsed, dict):
                        content = parsed.get("content", [])
                        if isinstance(content, list) and content and isinstance(content[0], dict):
                            fmt_input = content[0].get("text", result)
                except (ValueError, TypeError):
                    pass
                formatted = await _run_skill_script(skill, "format_tickets.py", fmt_input)
                if formatted:
                    result = formatted
            messages.append(ToolMessage(content=result, tool_call_id=tc.get("id")))
            tool_texts.append(result)

    # 生成最终回复：需要审核的意图禁用流式（走 human_review interrupt 闭环），否则流式
    content = ""
    q = get_stream_queue()
    if turn_needs_review is None:
        needs_review = review_intents and skill.intent in review_intents
    else:
        needs_review = turn_needs_review
    if needs_review:
        # 禁用流式：完整生成，交给 human_review 节点 interrupt 挂起
        try:
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"[{name}] 生成失败: {e}")
            content = ""
    else:
        # 正常流式，过程中同步做规则护栏
        try:
            from ..harness.guardrails import stream_rule_hit
            async for chunk in llm.astream(messages):
                token = chunk.content
                if isinstance(token, str) and token:
                    content += token
                    if stream_rule_hit(content):
                        if q is not None:
                            await q.put(("__blocked__", ""))
                        return "", tool_texts
                    if q is not None:
                        await q.put((name, token))
        except Exception as e:
            # 流式响应偶发失败，回退非流式完整生成
            logger.warning(f"[{name}] 流式生成失败，回退非流式: {e}")
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            if q is not None:
                await q.put((name, content))

    return content, tool_texts


def make_agent_node(llm, tools, skill, memory=None, memory_llm=None, review_intents=None, agent_to_intent=None,
                    use_format_script=True):
    """把专职 Agent 包装成 Supervisor 图里的节点（自定义 ReAct + 流式生成）。"""
    async def agent_node(state) -> Command:
        name = skill.name
        # 注入长期记忆上下文（只影响本轮，不写入共享 messages）
        memory_context = state.get("memory_context", "")
        input_messages = list(state["messages"])
        prefix = []
        summary = state.get("conversation_summary", "")
        if summary:
            prefix.append(SystemMessage(content="## 对话历史摘要\n" + summary))
        if memory_context:
            prefix.append(SystemMessage(content="## 用户长期记忆\n" + memory_context))

        # 整轮级审核判断：本轮派发的任一 Agent 命中审核意图，则本轮所有 Worker 都禁用流式，
        # 统一走 human_review 闭环。
        turn_needs_review = None
        if agent_to_intent is not None:
            turn_needs_review = bool(review_intents) and any(
                agent_to_intent.get(a) in review_intents
                for a in state.get("agents_to_dispatch", [])
            )

        try:
            content, tool_texts = await _run_react(llm, tools, skill, prefix, input_messages, name, review_intents, turn_needs_review, use_format_script)
            logger.info(f"[{name}] 完成，回复长度 {len(content)}")

            # 记忆固化：后台异步处理
            if memory and memory_llm:
                user_text = _latest_user_text(state.get("messages", []))
                asyncio.create_task(_async_remember(memory, memory_llm, user_text, content, name))
        except Exception as e:
            # 单 Worker 失败不影响其他并行 Worker，返回兜底话术
            logger.error(f"[{name}] 执行失败: {e}")
            content = f"[{name}] 处理失败，请稍后重试"
            tool_texts = []

        # 只回写本 Worker 的 delta，交由 state 的 _accumulate_or_reset reducer 合并
        return Command(
            update={
                "messages": [HumanMessage(content=content, name=name)],
                "worker_results": [{"agent": name, "response": content}],
                "tool_results": tool_texts,
            },
            goto="join",
        )
    return agent_node
