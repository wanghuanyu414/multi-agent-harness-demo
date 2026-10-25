"""为 function calling 已确定的工具生成显式执行计划与推理。"""
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState
from ..prompts import PLANNER_PROMPT
from ..langsmith_client import get_prompt
from ..llm_utils import ainvoke_with_retry, parse_llm_json

logger = logging.getLogger("harness.planner")


async def _llm_generate_plan(llm: ChatOpenAI, tool_calls: list, user_query: str) -> dict | None:
    """基于 function calling 已确定的 tool_calls，生成显式执行计划（含整体推理）"""
    prompt = get_prompt("planner", PLANNER_PROMPT).format(
        tool_calls=json.dumps(tool_calls, ensure_ascii=False),
        user_query=user_query,
    )
    for _ in range(3):
        try:
            response = await ainvoke_with_retry(
                llm,
                [SystemMessage(content=prompt), HumanMessage(content="请生成执行计划")],
            )
            plan = parse_llm_json(response.content)
            if isinstance(plan, dict) and "steps" in plan:
                return plan
        except Exception as e:
            logger.warning(f"LLM 规划失败: {e}")
    return None


async def planner_node(state: AgentState, llm: ChatOpenAI, mcp_client=None,
                       config: dict | None = None) -> dict[str, Any]:
    """为 function calling 已确定的工具生成显式计划与推理。

    失败时回退到 tool_calls 原样 + 简单顺序计划。
    """
    intent = state.get("intent", "chat")
    tool_calls = state.get("tool_calls", [])
    if intent == "chat" or not tool_calls:
        return {"intent": intent}

    user_query = ""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        user_query = last.content if hasattr(last, "content") else str(last)

    plan = await _llm_generate_plan(llm, tool_calls, user_query)

    if not plan:
        return {
            "tool_calls": tool_calls,
            "execution_plan": [
                {"step": i, "tool": tc.get("name", ""), "reason": ""}
                for i, tc in enumerate(tool_calls)
            ],
            "current_step": 0,
            "plan_reasoning": "",
            "intent": intent,
        }

    steps = plan.get("steps", [])
    reasoning = plan.get("reasoning", "")
    logger.info(f"规划: {len(steps)} 步 | 推理: {reasoning}")

    return {
        "tool_calls": tool_calls,
        "execution_plan": steps,
        "current_step": 0,
        "plan_reasoning": reasoning,
        "intent": intent,
    }
