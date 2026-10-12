"""动态规划 —— LLM 驱动的多步任务分解与推理（CoT）"""
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState
from ..mcp_client import MCPClient
from ..nodes import route_tool_node
from ..prompts import PLANNER_PROMPT
from ..langsmith_client import get_prompt
from ..llm_utils import ainvoke_with_retry, parse_llm_json

logger = logging.getLogger("harness.planner")

async def _llm_generate_plan(
    llm: ChatOpenAI,
    intent: str,
    params: dict,
    tools_desc: str,
    max_steps: int,
    user_query: str,
) -> dict | None:
    """使用 LLM 生成多步执行计划（含整体推理）"""
    prompt = get_prompt("planner", PLANNER_PROMPT).format(
        tools_description=tools_desc,
        intent=intent,
        params=json.dumps(params, ensure_ascii=False),
        max_steps=max_steps,
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


def _validate_plan(plan: dict | None, mcp_client: MCPClient) -> bool:
    """验证多步计划是否合法：每个步骤的工具存在于注册表、参数必填字段满足"""
    if not plan or "steps" not in plan:
        return False
    steps = plan.get("steps", [])
    if not steps or not isinstance(steps, list):
        return False
    for step in steps:
        server = step.get("tool_server", "")
        tool_name = step.get("tool_name", "")
        if not server or not tool_name:
            return False
        # 检查工具是否在注册表中
        tool = mcp_client.find_tool(server, tool_name)
        if not tool:
            logger.warning(f"计划验证失败: 工具不存在 server={server}, tool={tool_name}")
            return False
    return True


def _static_fallback(state: AgentState) -> dict[str, Any]:
    """回退到静态路由逻辑"""
    logger.info("回退到静态路由 route_tool_node")
    return route_tool_node(state)


async def planner_node(
    state: AgentState,
    llm: ChatOpenAI,
    mcp_client: MCPClient,
    config: dict,
) -> dict[str, Any]:
    """动态规划节点 —— LLM 生成多步执行计划（含 CoT 推理），替代硬编码路由

    失败时回退到静态 route_tool_node
    """
    intent = state.get("intent", "chat")
    # 优先用 normalize_params 子图产出的干净参数；日期用解析后的 YYYY-MM-DD
    params = dict(state.get("normalized_params") or state.get("extracted_params", {}))
    if state.get("date_parsed"):
        params["date_parsed"] = state["date_parsed"]

    if intent == "chat":
        return {"intent": "chat"}

    # 原始用户请求（供规划器拆解组合请求，如"查天气+查车票"）
    user_query = ""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        user_query = last.content if hasattr(last, "content") else str(last)

    max_steps = config.get("max_steps", 3)

    # 获取工具描述
    tools_desc = mcp_client.get_all_tools_description()
    if not tools_desc.strip():
        logger.warning("无可用工具描述，回退到静态路由")
        if config.get("fallback_to_static", True):
            return _static_fallback(state)
        return {"error": "无可用工具", "intent": intent}

    # LLM 生成多步计划
    plan = await _llm_generate_plan(llm, intent, params, tools_desc, max_steps, user_query)

    if not _validate_plan(plan, mcp_client):
        logger.warning("LLM 计划无效，回退到静态路由")
        if config.get("fallback_to_static", True):
            return _static_fallback(state)
        return {"error": "无法生成有效的执行计划，请重试", "intent": intent}

    steps = plan.get("steps", [])
    reasoning = plan.get("reasoning", "")
    logger.info(f"动态规划: {len(steps)} 步 | 推理: {reasoning[:80]}")

    result: dict[str, Any] = {
        "execution_plan": steps,
        "current_step": 0,
        "plan_reasoning": reasoning,
        "intent": intent,
    }
    # 兼容单步旧逻辑：同时设置第一步的工具信息
    if steps:
        result["mcp_server"] = steps[0]["tool_server"]
        result["tool_name"] = steps[0]["tool_name"]
        result["tool_args"] = steps[0].get("tool_args", {})
    return result
