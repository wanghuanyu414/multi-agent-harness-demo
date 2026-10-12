"""参数规范化子图 —— 把 LLM 抽出的参数规范化为可直接执行的工具参数。"""
import logging
from datetime import date
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from ..date_utils import parse_date, extract_date_from_message
from ..builtin_tools import normalize_city_name, KNOWN_CITIES

logger = logging.getLogger(__name__)


class ParamNormalizeState(TypedDict):
    intent: str
    extracted_params: dict    # 输入：LLM 抽出的原始参数
    user_text: str            # 输入：最新用户原文（日期兜底解析）
    normalized_params: dict   # 输出：规范化后的参数
    date_parsed: str          # 输出：YYYY-MM-DD 或 ""
    tool_args: dict           # 输出：单步静态路径的工具参数
    error: str                # 输出：缺参/非法错误


# ============================================================
# L2 工具
# ============================================================

class ResolveDateArgs(BaseModel):
    date: str  # YYYY-MM-DD 绝对日期


def _resolve_date_sink(date: str) -> str:
    return date


RESOLVE_DATE_TOOL = StructuredTool.from_function(
    func=_resolve_date_sink,
    name="parse-date",
    description="把用户给的相对/自然语言日期解析为 YYYY-MM-DD 绝对日期后，调用本工具提交",
    args_schema=ResolveDateArgs,
)


class ResolveCityArgs(BaseModel):
    city: str  # 标准城市名


def _resolve_city_sink(city: str) -> str:
    return city


RESOLVE_CITY_TOOL = StructuredTool.from_function(
    func=_resolve_city_sink,
    name="resolve-city",
    description="把用户说的城市/车站归一化为标准城市名（如「帝都」→北京、「北京南站」→北京）",
    args_schema=ResolveCityArgs,
)


def _extract_tool_args(resp, tool_name: str) -> dict:
    """从 LLM 响应中提取指定工具的入参（兼容 dict / ToolCall 对象两种形态）"""
    for tc in getattr(resp, "tool_calls", None) or []:
        if isinstance(tc, dict):
            if tc.get("name") == tool_name:
                return tc.get("args") or {}
        else:
            if getattr(tc, "name", "") == tool_name:
                return getattr(tc, "args", None) or {}
    return {}


async def _llm_resolve_date(llm, date_text: str) -> str:
    """L2 日期解析：强制调用 parse-date 工具。失败返回 ""。"""
    try:
        forced = llm.bind_tools(
            [RESOLVE_DATE_TOOL],
            tool_choice={"type": "function", "function": {"name": "parse-date"}},
        )
        resp = await forced.ainvoke([
            SystemMessage(content=(
                f"今天是 {date.today().isoformat()}。"
                "请把用户的日期表达解析为 YYYY-MM-DD 绝对日期，并调用 parse-date 工具提交。"
            )),
            HumanMessage(content=date_text),
        ])
        raw = _extract_tool_args(resp, "parse-date").get("date")
        if raw:
            return parse_date(str(raw)) or str(raw)
    except Exception as e:
        logger.warning("L2 日期解析失败，回退 L0: %s", type(e).__name__)
    return ""


async def _llm_resolve_city(llm, city_text: str) -> str:
    """L2 城市归一化/车站消歧：强制调用 resolve-city 工具。失败返回 ""。"""
    try:
        forced = llm.bind_tools(
            [RESOLVE_CITY_TOOL],
            tool_choice={"type": "function", "function": {"name": "resolve-city"}},
        )
        known = "、".join(sorted(KNOWN_CITIES))
        resp = await forced.ainvoke([
            SystemMessage(content=f"把用户说的城市/车站归一化为标准城市名。已知城市：{known}"),
            HumanMessage(content=city_text),
        ])
        city = _extract_tool_args(resp, "resolve-city").get("city")
        if city and city in KNOWN_CITIES:
            return city
    except Exception as e:
        logger.warning("L2 城市解析失败，回退 L0: %s", type(e).__name__)
    return ""


# ============================================================
# 节点（L0 快路径 → L2 模糊 → L0 兜底）
# ============================================================

async def parse_date_node(state: ParamNormalizeState, llm=None) -> dict:
    """解析出发日期"""
    intent = state.get("intent", "chat")
    if intent != "ticket":
        return {"date_parsed": ""}

    date_text = state.get("extracted_params", {}).get("date")
    if date_text:
        parsed = parse_date(str(date_text))
        if parsed:
            return {"date_parsed": parsed}

    user_text = state.get("user_text", "")
    parsed = extract_date_from_message(user_text)
    if parsed:
        return {"date_parsed": parsed}

    # L2：模糊日期（下周三/月底等）
    source = date_text or user_text
    if source and llm is not None:
        parsed = await _llm_resolve_date(llm, str(source))
        if parsed:
            return {"date_parsed": parsed}

    return {"date_parsed": ""}


async def _normalize_one_city(raw: str, llm=None) -> str:
    """L0 去后缀/英文转中文 → L2 别名/车站消歧"""
    normalized = normalize_city_name(raw)
    if normalized in KNOWN_CITIES:
        return normalized
    if llm is not None:
        resolved = await _llm_resolve_city(llm, raw)
        if resolved:
            return resolved
    return normalized


async def normalize_city_node(state: ParamNormalizeState, llm=None) -> dict:
    """城市归一化：英文→中文、去「市/县」后缀、别名/车站消歧"""
    intent = state.get("intent", "chat")
    params = dict(state.get("extracted_params", {}))

    if intent == "weather":
        if params.get("city"):
            params["city"] = await _normalize_one_city(params["city"], llm)
    elif intent == "ticket":
        for key in ("from_city", "to_city"):
            if params.get(key):
                params[key] = await _normalize_one_city(params[key], llm)

    return {"normalized_params": params}


def validate_required_node(state: ParamNormalizeState) -> dict:
    """缺参校验：weather 需 city；ticket 需 from/to/date"""
    intent = state.get("intent", "chat")
    params = state.get("normalized_params", {})

    if intent == "weather":
        if not params.get("city"):
            return {"error": "请告诉我您想查询哪个城市的天气？例如：北京天气怎么样"}
    elif intent == "ticket":
        missing = []
        if not params.get("from_city"):
            missing.append("出发城市")
        if not params.get("to_city"):
            missing.append("目的城市")
        if not state.get("date_parsed"):
            missing.append("出发日期")
        if missing:
            return {"error": f"请提供以下信息：{'、'.join(missing)}。\n例如：明天从北京到上海的高铁"}

    return {"error": ""}


def build_tool_args_node(state: ParamNormalizeState) -> dict:
    """组装最终工具参数（单步静态路径用）"""
    intent = state.get("intent", "chat")
    if state.get("error"):
        return {"tool_args": {}}

    params = state.get("normalized_params", {})
    if intent == "weather":
        return {"tool_args": {"city": params["city"]}}
    if intent == "ticket":
        return {"tool_args": {
            "train_date": state.get("date_parsed", ""),
            "from_station": params["from_city"],
            "to_station": params["to_city"],
        }}
    return {"tool_args": {}}


def build_param_normalize_graph(llm=None):
    """构建参数规范化子图（编译后的 StateGraph，可作为父图节点）。

    llm 为 None 时退化为纯 L0（完全确定性，无 LLM 调用）。
    """
    async def _parse_date(state):
        return await parse_date_node(state, llm)

    async def _normalize_city(state):
        return await normalize_city_node(state, llm)

    workflow = StateGraph(ParamNormalizeState)
    workflow.add_node("parse_date", _parse_date)
    workflow.add_node("normalize_city", _normalize_city)
    workflow.add_node("validate_required", validate_required_node)
    workflow.add_node("build_tool_args", build_tool_args_node)

    workflow.add_edge(START, "parse_date")
    workflow.add_edge("parse_date", "normalize_city")
    workflow.add_edge("normalize_city", "validate_required")
    workflow.add_edge("validate_required", "build_tool_args")
    workflow.add_edge("build_tool_args", END)

    return workflow.compile()
