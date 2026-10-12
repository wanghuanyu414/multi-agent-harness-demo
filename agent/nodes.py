"""LangGraph 工作流节点函数"""
import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .state import AgentState
from .mcp_client import MCPClient
from .tool_errors import ToolCallError
from .date_utils import extract_date_from_message
from .prompts import INTENT_PROMPT, RESPONSE_PROMPT
from .langsmith_client import get_prompt
from .llm_utils import ainvoke_with_retry, parse_llm_json
from .memory import extract_facts

logger = logging.getLogger(__name__)


def format_raw_result(intent: str, mcp_result: dict) -> str:
    """当 LLM 不可用时，直接从 MCP 结果生成可读回复"""
    # MCP 标准响应: {"content": [{"type": "text", "text": "..."}]}
    content_items = mcp_result.get("content", [])
    if content_items and isinstance(content_items, list):
        text = content_items[0].get("text", "")
        if text:
            if intent == "ticket":
                emoji = "🚄"
                title = "车票查询结果"
            elif intent == "weather":
                emoji = "🌤️"
                title = "天气查询结果"
            else:
                emoji = "📋"
                title = "查询结果"
            return f"{emoji} **{title}**\n\n{text}"
    # 回退：JSON dump
    result_str = json.dumps(mcp_result, ensure_ascii=False, indent=2)
    return f"📋 查询结果（原始数据）：\n\n```json\n{result_str[:3000]}\n```"

# ============================================================
# 意图分析
# ============================================================

# 常见中国城市列表（用于关键词匹配）
_CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆", "武汉",
    "西安", "郑州", "苏州", "天津", "长沙", "东莞", "沈阳", "青岛", "合肥",
    "佛山", "大连", "福州", "厦门", "哈尔滨", "济南", "温州", "南宁", "长春",
    "泉州", "贵阳", "南昌", "金华", "常州", "惠州", "嘉兴", "徐州", "南通",
    "太原", "珠海", "中山", "保定", "兰州", "台州", "绍兴", "烟台", "廊坊",
    "乌鲁木齐", "昆明", "汕头", "洛阳", "呼和浩特", "拉萨", "银川", "海口",
    "三亚", "桂林", "丽江", "张家界", "黄山",
]

# 天气关键词
_WEATHER_KEYWORDS = ["天气", "气温", "温度", "下雨", "晴天", "阴天", "刮风",
                      "冷", "热", "凉快", "暖和", "雾霾", "空气质量", "湿度",
                      "风力", "紫外线", "穿衣", "带伞", "会不会下雨"]

# 车票关键词
_TICKET_KEYWORDS = ["火车票", "高铁", "动车", "车票", "订票", "购票", "票",
                    "列车", "车次", "二等座", "一等座", "商务座", "硬座", "硬卧",
                    "软卧", "余票", "12306", "火车", "铁路", "直达", "特快"]


def _keyword_intent(text: str) -> dict:
    """基于关键词的意图识别（LLM 不可用时的降级方案）"""
    # 检测车票关键词
    ticket_score = sum(1 for kw in _TICKET_KEYWORDS if kw in text)
    # 检测天气关键词
    weather_score = sum(1 for kw in _WEATHER_KEYWORDS if kw in text)

    # 提取城市名（按在文本中出现的位置排序）
    cities_found = sorted(
        [c for c in _CITIES if c in text],
        key=lambda c: text.index(c)
    )
    date_found = extract_date_from_message(text)

    # 判断意图
    if ticket_score > 0 and len(cities_found) >= 2:
        return {
            "intent": "ticket",
            "city": None,
            "from_city": cities_found[0],
            "to_city": cities_found[1] if len(cities_found) > 1 else None,
            "date": date_found or (extract_date_from_message(text) if not date_found else date_found),
        }
    elif weather_score > 0 and cities_found:
        return {
            "intent": "weather",
            "city": cities_found[0],
            "from_city": None,
            "to_city": None,
            "date": None,
        }
    elif len(cities_found) >= 2:
        return {
            "intent": "ticket",
            "city": None,
            "from_city": cities_found[0],
            "to_city": cities_found[1],
            "date": date_found,
        }
    elif cities_found and ("天气" in text or "怎么样" in text):
        return {
            "intent": "weather",
            "city": cities_found[0],
            "from_city": None, "to_city": None, "date": None,
        }
    else:
        return {"intent": "chat", "city": None, "from_city": None, "to_city": None, "date": None}


async def analyze_intent_node(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """意图分析节点 —— 使用 LLM 识别用户意图并提取参数（LLM 不可用时降级为关键词匹配）"""
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "chat", "extracted_params": {}, "user_text": ""}

    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    logger.info(f"正在分析意图: {user_text[:100]}")

    result = None
    # 尝试 LLM
    try:
        response = await ainvoke_with_retry(llm, [
            SystemMessage(content=get_prompt("intent", INTENT_PROMPT)),
            HumanMessage(content=user_text),
        ])
        result = parse_llm_json(response.content)
        if not isinstance(result, dict):
            raise ValueError("LLM 意图输出不是有效 JSON 对象")
    except Exception as e:
        logger.warning(f"LLM 意图分析失败，使用关键词降级: {e}")
        result = _keyword_intent(user_text)

    intent = result.get("intent", "chat")
    params: dict[str, str | None] = {
        "city": result.get("city"),
        "from_city": result.get("from_city"),
        "to_city": result.get("to_city"),
        "date": result.get("date"),
    }

    # 日期解析/城市归一化/缺参校验统一交给 normalize_params 子图处理
    logger.info(f"意图: {intent}, 参数: {params}")
    return {"intent": intent, "extracted_params": params, "user_text": user_text}


# ============================================================
# 工具路由
# ============================================================

def route_tool_node(state: AgentState) -> dict[str, Any]:
    """工具路由节点 —— 按意图映射 MCP 服务器与工具名。"""
    intent = state.get("intent", "chat")
    if intent == "weather":
        return {"mcp_server": "weather", "tool_name": "get_weather"}
    if intent == "ticket":
        return {"mcp_server": "12306", "tool_name": "query-tickets"}
    return {"intent": "chat"}


# ============================================================
# 工具执行
# ============================================================

async def execute_tool_node(state: AgentState, mcp_client: MCPClient) -> dict[str, Any]:
    """工具执行节点 —— 调用 MCP 工具（支持多步执行计划，逐步骤调用并累积结果）"""
    intent = state.get("intent", "chat")

    # 多步执行：从 execution_plan 读取当前步骤
    plan = state.get("execution_plan", [])
    current_step = state.get("current_step", 0)
    if plan and current_step < len(plan):
        step = plan[current_step]
        mcp_server = step.get("tool_server", "")
        tool_name_hint = step.get("tool_name", "")
        tool_args = step.get("tool_args", {})
    else:
        mcp_server = state.get("mcp_server", "")
        tool_name_hint = state.get("tool_name", "")
        tool_args = state.get("tool_args", {})

    if intent == "chat" or not mcp_server:
        return {}

    logger.info(f"调用 MCP: server={mcp_server}, tool={tool_name_hint}, args={tool_args}")

    mcp_result: dict = {}
    try:
        # 检查服务器是否在运行
        if not mcp_client.is_server_running(mcp_server):
            actual_error = mcp_client.get_startup_error(mcp_server)
            if actual_error:
                error_msg = f"🚫 {mcp_server} 服务未启动。原因: {actual_error}"
            else:
                error_msg = (
                    f"🚫 {mcp_server} 服务未启动，请检查服务配置或稍后重试。"
                    f" [活跃服务器: {list(mcp_client._servers.keys()) or '无'}]"
                )
            mcp_result = {"error": error_msg}
        else:
            # 查找匹配的工具
            tool = mcp_client.find_tool(mcp_server, tool_name_hint)
            if not tool:
                mcp_result = {
                    "error": f"服务器 {mcp_server} 上未找到工具 '{tool_name_hint}'，可用工具已列出"
                }
            else:
                actual_tool_name = tool["name"]
                mcp_result = await mcp_client.call_tool(mcp_server, actual_tool_name, tool_args)
                # 间接注入防护：替换为安全标记
                from .harness.guardrails import sanitize_tool_output
                mcp_result, _ = sanitize_tool_output(mcp_result)
    except ToolCallError as e:
        logger.error(f"工具调用失败 [{e.code}]: {e.detail}", exc_info=True)
        mcp_result = {"error": e.user_message}
    except Exception as e:
        logger.error(f"工具调用失败: {e}", exc_info=True)
        mcp_result = {"error": f"服务暂时不可用，请稍后重试（{type(e).__name__}）"}

    # 累积步骤结果 + 推进到下一步
    step_results = list(state.get("step_results", []))
    step_results.append({
        "step": current_step,
        "tool_name": tool_name_hint,
        "result": mcp_result,
    })
    return {
        "mcp_result": mcp_result,
        "intent": intent,
        "step_results": step_results,
        "current_step": current_step + 1,
    }


# ============================================================
# 响应生成
# ============================================================

def _latest_user_text(messages) -> str:
    """取最新一条真实用户消息（不含 Worker 回写的带 name 消息）"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and not getattr(m, "name", None):
            return m.content if hasattr(m, "content") else str(m)
    return ""


async def generate_response_node(state: AgentState, llm: ChatOpenAI, memory=None) -> dict[str, Any]:
    """响应生成节点 —— LLM 整合结果生成最终回复（含长期记忆召回与固化）"""
    intent = state.get("intent", "chat")
    mcp_result = state.get("mcp_result", {})
    error = state.get("error", "")
    messages = state.get("messages", [])
    user_text = _latest_user_text(messages)

    # 召回长期记忆上下文（失败不影响主流程）
    memory_context = ""
    if memory and user_text:
        try:
            memories = memory.recall(user_text, top_k=5)
            memory_context = memory.to_context(memories)
        except Exception as e:
            logger.warning(f"记忆召回失败: {e}")

    final_response: str | None = None

    # 如果有参数缺失错误，直接返回
    if error:
        final_response = f"😊 {error}"

    # 一般对话，直接用 LLM 回复
    elif intent == "chat":
        if messages:
            try:
                sys_msgs = [SystemMessage(
                    content="你是一个友好的中文旅行助手，请用中文回复用户，适当使用表情符号。"
                )]
                if memory_context:
                    sys_msgs.append(SystemMessage(content="## 用户长期记忆\n" + memory_context))
                response = await ainvoke_with_retry(llm, [*sys_msgs, *list(messages[-4:])])
                if response.content and len(response.content.strip()) > 2:
                    final_response = response.content
                else:
                    final_response = "你好！我是旅行助手，可以帮你查天气、查火车票。有什么需要帮助的吗？"
            except Exception as e:
                logger.error(f"LLM 聊天失败: {type(e).__name__}: {e}")
                final_response = "你好！我是旅行助手，可以帮你查天气、查火车票。有什么需要帮助的吗？"

    else:
        # 多步执行：聚合所有步骤结果
        step_results = state.get("step_results", [])
        if len(step_results) > 1:
            mcp_result = {
                "steps": [
                    {"step": s.get("step"), "tool": s.get("tool_name"), "result": s.get("result")}
                    for s in step_results
                ]
            }
        elif step_results:
            mcp_result = step_results[0].get("result", mcp_result)

        # 工具调用结果，用 LLM 格式化
        if mcp_result:
            if "error" in mcp_result:
                error_msg = mcp_result["error"]
                final_response = f"😅 抱歉，查询时遇到了问题：{error_msg}\n请稍后重试或检查输入信息是否正确。"
            else:
                result_str = json.dumps(mcp_result, ensure_ascii=False, indent=2)
                if result_str in ("{}", "[]", '""', "null") or len(result_str) < 10:
                    final_response = "😅 抱歉，没有查询到相关结果。请确认输入信息是否正确，或者换个方式试试～"
                else:
                    try:
                        response = await ainvoke_with_retry(llm, [
                            SystemMessage(content=get_prompt("response", RESPONSE_PROMPT).replace(
                                "{intent}", f"{intent}查询"
                            ).replace(
                                "{mcp_result}", result_str[:2000]
                            )),
                            HumanMessage(content="请根据以上信息生成友好的中文回复"),
                        ])
                        if response.content and len(response.content.strip()) > 5:
                            final_response = response.content
                        else:
                            # LLM 返回空内容，降级为原始数据展示
                            final_response = format_raw_result(intent, mcp_result)
                    except Exception as e:
                        logger.error(f"LLM 格式化失败: {type(e).__name__}: {e}")
                        final_response = format_raw_result(intent, mcp_result)

    if final_response is None:
        final_response = "😊 有什么我可以帮你的吗？"

    # 记忆固化：从本轮对话提取值得长期记住的事实（失败不影响主流程）
    if memory and user_text:
        try:
            facts = await extract_facts(llm, user_text, final_response)
            for f in facts:
                memory.remember(f, kind="fact")
        except Exception as e:
            logger.warning(f"记忆固化失败: {e}")

    return {"final_response": final_response, "memory_context": memory_context}