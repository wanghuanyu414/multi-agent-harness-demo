"""反思循环 —— 评估工具执行结果并决定是否重试"""
import json
import logging
from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..state import AgentState
from ..prompts import REFLECTION_QUALITY_PROMPT
from ..langsmith_client import get_prompt

logger = logging.getLogger("harness.reflection")

# ============================================================
# 参数调整策略（按重试次数索引）
# ============================================================

_CITY_EN_MAP: dict[str, str] = {
    "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou", "深圳": "Shenzhen",
    "杭州": "Hangzhou", "南京": "Nanjing", "成都": "Chengdu", "重庆": "Chongqing",
    "武汉": "Wuhan", "西安": "Xi'an", "郑州": "Zhengzhou", "苏州": "Suzhou",
    "天津": "Tianjin", "长沙": "Changsha", "青岛": "Qingdao", "厦门": "Xiamen",
    "大连": "Dalian", "昆明": "Kunming", "三亚": "Sanya", "桂林": "Guilin",
}

_EN_CITY_REVERSE: dict[str, str] = {v: k for k, v in _CITY_EN_MAP.items()}


def _adjust_weather_args(tool_args: dict, retry_count: int) -> dict:
    """调整天气查询参数"""
    args = deepcopy(tool_args)
    city = args.get("city", "")

    strategies = [
        # 第1次：追加「市」
        lambda c: c + "市" if not c.endswith("市") and not c.endswith("县") else c,
        # 第2次：尝试英文名
        lambda c: _CITY_EN_MAP.get(c, c),
        # 第3次：去掉后缀，用纯城市名
        lambda c: c.rstrip("市县城"),
    ]

    if retry_count < len(strategies):
        new_city = strategies[retry_count](city)
        if new_city != city:
            logger.info(f"反思调整 weather city: {city} → {new_city}")
            args["city"] = new_city

    return args


def _adjust_ticket_args(tool_args: dict, retry_count: int) -> dict:
    """调整车票查询参数"""
    args = deepcopy(tool_args)

    strategies = [
        # 第1次：追加「站」
        lambda a: {
            **a, "from_station": a.get("from_station", "") + ("站" if a.get("from_station", "") else ""),
            "to_station": a.get("to_station", "") + ("站" if a.get("to_station", "") else ""),
        },
        # 第2次：交换出发/目的
        lambda a: {
            **a,
            "from_station": a.get("to_station", a.get("from_station", "")),
            "to_station": a.get("from_station", a.get("to_station", "")),
        },
        # 第3次：追加「市」
        lambda a: {
            **a,
            "from_station": a.get("from_station", "") + ("市" if a.get("from_station", "") and not a["from_station"].endswith("市") else ""),
            "to_station": a.get("to_station", "") + ("市" if a.get("to_station", "") and not a["to_station"].endswith("市") else ""),
        },
    ]

    if retry_count < len(strategies):
        new_args = strategies[retry_count](args)
        logger.info(f"反思调整 ticket args: {args} → {new_args}")
        args = new_args

    return args


_ADJUSTERS: dict[str, callable] = {
    "weather": _adjust_weather_args,
    "ticket": _adjust_ticket_args,
}


# ============================================================
# 结果评估
# ============================================================

def _evaluate_result(mcp_result: dict, intent: str, config: dict) -> dict:
    """评估 MCP 工具结果是否令人满意

    Returns:
        {"needs_retry": bool, "reason": str}
    """
    retry_on = config.get("retry_on", {})

    # 检查 1: 工具返回了 error
    if retry_on.get("tool_error") and isinstance(mcp_result, dict):
        if "error" in mcp_result:
            return {"needs_retry": True, "reason": f"工具返回错误: {mcp_result['error'][:100]}"}
        # MCP content 中也可能包含错误文本
        content = mcp_result.get("content", [])
        if isinstance(content, list) and len(content) > 0:
            text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
            if text and ("error" in text.lower() or "失败" in text or "错误" in text):
                return {"needs_retry": True, "reason": f"工具结果包含错误: {text[:100]}"}

    # 检查 2: 结果为空
    if retry_on.get("empty_result"):
        if not mcp_result or mcp_result == {}:
            return {"needs_retry": True, "reason": "工具返回空结果"}
        # MCP 标准格式: {"content": [{"type": "text", "text": "..."}]}
        content = mcp_result.get("content", [])
        if isinstance(content, list) and len(content) == 0:
            # 可能是非 MCP 格式的结果（如内置工具），检查是否有实际数据
            if _has_meaningful_data(mcp_result):
                return {"needs_retry": False, "reason": "结果包含有效数据"}
            return {"needs_retry": True, "reason": "工具返回空 content"}
        if isinstance(content, list) and len(content) == 1:
            item = content[0]
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            if not text or len(text.strip()) < 10:
                return {"needs_retry": True, "reason": "工具返回内容过短或无意义"}

    return {"needs_retry": False, "reason": "结果似乎满意"}


def _has_meaningful_data(result: dict) -> bool:
    """检查非 MCP 格式结果是否包含有意义的数据"""
    if not isinstance(result, dict):
        return False
    # 有实际数据键（排除元数据键如 error）
    data_keys = [k for k in result.keys() if k not in ("error", "status", "message")]
    if not data_keys:
        return False
    # 至少有一个非空值
    return any(v for v in (result.get(k) for k in data_keys) if v not in (None, "", [], {}))


# ============================================================
# 反思节点
# ============================================================

class QualityVerdict(BaseModel):
    """LLM 质量评估结果"""
    needs_retry: bool
    reason: str = ""


def _latest_user_text(messages) -> str:
    """取最新一条真实用户消息"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and not getattr(m, "name", None):
            return m.content if hasattr(m, "content") else str(m)
    return ""


async def _evaluate_quality(llm, mcp_result, intent: str, user_query: str) -> dict:
    """LLM 评估工具结果质量（low_quality 开关的真正实现）。失败时视为可接受。"""
    try:
        judge_llm = llm.model_copy(update={"temperature": 0.0})
    except Exception:
        judge_llm = llm
    prompt = get_prompt("reflection-quality", REFLECTION_QUALITY_PROMPT).format(
        intent=intent,
        user_query=user_query[:200],
        mcp_result=json.dumps(mcp_result, ensure_ascii=False)[:2000],
    )
    try:
        verdict = await judge_llm.with_structured_output(QualityVerdict, method="function_calling").ainvoke([
            SystemMessage(content="你是工具结果质量评估员，判断结果是否满足用户请求。"),
            HumanMessage(content=prompt),
        ])
        if verdict.needs_retry:
            return {"needs_retry": True, "reason": verdict.reason or "LLM 判定结果质量不足"}
    except Exception as e:
        logger.warning("LLM 质量评估失败，跳过: %s", e)
    return {"needs_retry": False, "reason": "质量可接受"}


async def reflect_node(state: AgentState, config: dict, llm=None) -> dict[str, Any]:
    """反思节点 —— 评估 execute_tool 结果并决定是否重试

    Args:
        llm: 可选的 ChatOpenAI，用于 low_quality 质量评估；None 则跳过
    """
    intent = state.get("intent", "chat")
    retry_count = state.get("retry_count", 0)
    max_retries = config.get("max_retries", state.get("max_retries", 3))
    tool_args = state.get("tool_args", {})

    # 非工具意图不需要反思
    if intent == "chat":
        return {"reflection_notes": "ok: 非工具意图，跳过反思"}

    # 评估：function calling 多工具结果逐个评估，任一空/错误才重试；
    # 无 tool_results 时回退旧单工具 mcp_result
    tool_results = state.get("tool_results", [])
    if tool_results:
        evaluation = {"needs_retry": False, "reason": "结果满意"}
        for tr in tool_results:
            result = tr.get("result", {}) if isinstance(tr, dict) else tr
            ev = _evaluate_result(result, intent, config)
            if ev["needs_retry"]:
                evaluation = ev
                break
    else:
        evaluation = _evaluate_result(state.get("mcp_result", {}), intent, config)

    # 低质量评估：正则只查"报错/空结果"，这里补"结果错了/不匹配"的语义判断
    if (not evaluation["needs_retry"]
            and config.get("retry_on", {}).get("low_quality")
            and llm is not None):
        user_query = _latest_user_text(state.get("messages", []))
        evaluation = await _evaluate_quality(llm, mcp_result, intent, user_query)

    if evaluation["needs_retry"] and retry_count < max_retries:
        new_retry_count = retry_count + 1
        adjuster = _ADJUSTERS.get(intent)
        new_args = adjuster(tool_args, retry_count) if adjuster else tool_args

        logger.info(f"反思: 需要重试 ({new_retry_count}/{max_retries})，原因: {evaluation['reason']}")
        return {
            "retry_count": new_retry_count,
            "max_retries": max_retries,
            "reflection_notes": f"retry: {evaluation['reason']}",
            "tool_args": new_args,
        }

    if evaluation["needs_retry"]:
        logger.warning(f"反思: 已达最大重试次数 ({max_retries})，停止重试")
        return {
            "reflection_notes": f"ok: 已达最大重试次数 ({max_retries})，最后原因: {evaluation['reason']}",
        }

    return {"reflection_notes": f"ok: {evaluation['reason']}"}
