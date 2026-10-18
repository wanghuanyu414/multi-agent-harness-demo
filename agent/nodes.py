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
from .date_utils import extract_date_from_message, parse_date
from .builtin_tools import normalize_city_name
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


async def analyze_intent_node(state: AgentState, llm: ChatOpenAI, tools: list,
                              tool_domain: dict | None = None) -> dict[str, Any]:
    """意图分析 + 工具调用"""
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "chat", "tool_calls": [], "user_text": ""}

    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    logger.info(f"正在分析意图: {user_text[:100]}")

    tool_calls: list = []
    try:
        response = await ainvoke_with_retry(
            llm.bind_tools(tools),
            [SystemMessage(content=get_prompt("intent", INTENT_PROMPT)),
             HumanMessage(content=user_text)],
        )
        tool_calls = list(getattr(response, "tool_calls", None) or [])
    except Exception as e:
        logger.warning(f"LLM 工具调用分析失败: {e}")
        tool_calls = []

    # 规范化参数（日期 → YYYY-MM-DD，城市/车站 → 标准名）
    normalized_calls = []
    for tc in tool_calls:
        args = dict(tc.get("args") or {})
        for key in ("date", "train_date"):
            if args.get(key):
                parsed = parse_date(str(args[key]))
                if parsed:
                    args[key] = parsed
        for key in ("city", "from_city", "to_city", "from_station", "to_station"):
            if args.get(key):
                args[key] = normalize_city_name(str(args[key]))
        normalized_calls.append({"name": tc.get("name"), "args": args, "id": tc.get("id")})

    # 从全部工具调用推导意图集合（多意图「天气+车票」→ ["weather", "ticket"]）
    # 用工具名 → domain 的确定性映射，不靠字符串猜
    intents: list = []
    for tc in normalized_calls:
        domain = (tool_domain or {}).get(tc.get("name") or "", "")
        if domain in ("weather", "ticket") and domain not in intents:
            intents.append(domain)

    intent = intents[0] if intents else "chat"
    logger.info(f"意图: {intents or ['chat']}, 工具调用数: {len(normalized_calls)}")
    return {"intent": intent, "intents": intents, "tool_calls": normalized_calls, "user_text": user_text}


# ============================================================
# 工具路由
# ============================================================

def route_tool_node(state: AgentState) -> dict[str, Any]:
    """工具路由节点 —— 仅按意图映射 MCP 服务器与工具名。

    工具参数（tool_args）与缺参校验已由 normalize_params 子图产出，
    本节点不再重复；error 由子图设置，父图 tool_router 据此分流。
    """
    intent = state.get("intent", "chat")
    if intent == "weather":
        return {"mcp_server": "weather", "tool_name": "get_weather"}
    if intent == "ticket":
        return {"mcp_server": "12306", "tool_name": "query-tickets"}
    return {"intent": "chat"}


# ============================================================
# 工具执行
# ============================================================

async def _format_ticket_result(result, use_format_script: bool = True) -> dict | None:
    """用 skills/ticket_agent/scripts/format_tickets.py 确定性格式化车票结果（第 3 级脚本）。

    use_format_script=False 时跳过脚本，原始 JSON 交给 LLM 流式格式化（streaming.ticket_format_script 控制）。
    """
    if not use_format_script:
        return None
    content = result.get("content", []) if isinstance(result, dict) else []
    text = content[0].get("text", "") if content else ""
    if not text:
        return None
    # 只有合法 JSON（query-tickets 真实返回）才格式化；测试/异常输入跳过
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return None
    from .multiagent.skills import SKILLS
    from .multiagent.skill_loader import run_skill_script
    ticket_skill = next((s for s in SKILLS if s.intent == "ticket"), None)
    if ticket_skill is None:
        return None
    md_path = getattr(ticket_skill, "_skill_md_path", None)
    if not md_path:
        return None
    formatted_text = await run_skill_script(md_path, "format_tickets.py", text)
    if not formatted_text:
        return None
    return {"content": [{"type": "text", "text": formatted_text}]}


async def execute_tool_node(state: AgentState, mcp_client: MCPClient, tool_server: dict,
                            use_format_script: bool = True) -> dict[str, Any]:
    """工具执行节点 —— 并行执行 function calling 产出的所有工具调用（支持多意图）。"""
    tool_calls = state.get("tool_calls", [])
    intent = state.get("intent", "chat")
    if not tool_calls:
        return {}

    async def run_one(tc) -> dict:
        name = tc.get("name") or ""
        args = tc.get("args") or {}
        server = tool_server.get(name, "builtin")
        logger.info(f"调用 MCP: server={server}, tool={name}, args={args}")
        try:
            result = await mcp_client.call_tool(server, name, args)
        except ToolCallError as e:
            result = {"error": e.user_message}
        except Exception as e:
            result = {"error": f"服务暂时不可用，请稍后重试（{type(e).__name__}）"}
        # 间接注入防护：工具返回内容含注入指令时替换为安全标记
        from .harness.guardrails import sanitize_tool_output
        result, _ = sanitize_tool_output(result)
        # 车票查询结果用 skill 脚本格式化（第 3 级）
        if name == "query-tickets":
            formatted = await _format_ticket_result(result, use_format_script)
            if formatted:
                result = formatted
        return {"tool_name": name, "result": result}

    tool_results = await asyncio.gather(*[run_one(tc) for tc in tool_calls])
    logger.info(f"工具执行完成: {len(tool_results)} 个")
    return {"tool_results": tool_results, "intent": intent}


# ============================================================
# 响应生成
# ============================================================

def _latest_user_text(messages) -> str:
    """取最新一条真实用户消息（不含 Worker 回写的带 name 消息）"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and not getattr(m, "name", None):
            return m.content if hasattr(m, "content") else str(m)
    return ""


async def _async_remember(memory, llm, user_text, content):
    """后台记忆固化：提取事实写入长期记忆。"""
    try:
        facts = await extract_facts(llm, user_text, content)
        for f in facts:
            memory.remember(f, kind="fact")
    except Exception as e:
        logger.warning(f"记忆固化失败: {e}")


async def _stream_generate(llm, messages, node_name="generate_response") -> str:
    """流式生成：llm.astream 逐个 token 写入请求级队列，返回累积的完整内容。

    流式过程中同步做规则输出护栏（PII 泄露等），命中立即中断流式，
    并发出 "__blocked__" 拦截标记（web 层据此让前端撤回已输出内容）。
    """
    from .metrics import get_stream_queue
    from .harness.guardrails import stream_rule_hit
    q = get_stream_queue()
    content = ""
    async for chunk in llm.astream(messages):
        token = chunk.content
        if isinstance(token, str) and token:
            content += token
            if stream_rule_hit(content):
                if q is not None:
                    await q.put(("__blocked__", ""))
                return ""
            if q is not None:
                await q.put((node_name, token))
    return content


async def _generate(llm, messages, stream=True) -> str:
    """生成回复；stream=False 时完整生成（需要审核的意图禁用流式，走 human_review interrupt 闭环）。"""
    if not stream:
        response = await llm.ainvoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    return await _stream_generate(llm, messages)


def _extract_ticket_table(tool_results: list) -> str | None:
    """从 tool_results 提取 query-tickets 的格式化表格（execute_tool 已用脚本确定性格式化）"""
    for tr in tool_results:
        if tr.get("tool_name") == "query-tickets":
            result = tr.get("result", {})
            content = result.get("content", []) if isinstance(result, dict) else []
            text = content[0].get("text", "") if content else ""
            if text and "|" in text:
                return text
    return None


def _extract_non_ticket_results(tool_results: list) -> list:
    """过滤掉 query-tickets，返回其他工具结果（天气等）"""
    return [r for r in tool_results if r.get("tool_name") != "query-tickets"]


async def generate_response_node(state: AgentState, llm: ChatOpenAI, memory=None, review_intents=None) -> dict[str, Any]:
    """响应生成节点 —— LLM 整合结果生成最终回复（含长期记忆召回与固化）"""
    intent = state.get("intent", "chat")
    intents = state.get("intents") or [intent]  # 多意图：任一命中审核即整轮禁用流式
    mcp_result = state.get("mcp_result", {})
    error = state.get("error", "")
    messages = state.get("messages", [])
    user_text = _latest_user_text(messages)

    # 需要审核的意图禁用流式（完整生成，走 human_review 节点 interrupt 闭环）
    needs_review = bool(review_intents and any(i in review_intents for i in intents))

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
                content = await _generate(llm, [*sys_msgs, *list(messages[-4:])], stream=not needs_review)
                if content and len(content.strip()) > 2:
                    final_response = content
                else:
                    final_response = "你好！我是旅行助手，可以帮你查天气、查火车票。有什么需要帮助的吗？"
            except Exception as e:
                logger.error(f"LLM 聊天失败: {type(e).__name__}: {e}")
                final_response = "你好！我是旅行助手，可以帮你查天气、查火车票。有什么需要帮助的吗？"

    else:
        # 并行工具执行：汇总所有工具结果
        tool_results = state.get("tool_results", [])
        if tool_results:
            errors = [r for r in tool_results
                      if isinstance(r.get("result"), dict) and "error" in r.get("result", {})]
            if errors and len(errors) == len(tool_results):
                final_response = f"😅 抱歉，查询时遇到了问题：{errors[0]['result']['error']}\n请稍后重试。"
            else:
                ticket_table = _extract_ticket_table(tool_results)
                if ticket_table:
                    # 车票表格：execute_tool 已用脚本确定性格式化，直接拼接，LLM 只生成建议
                    other_results = _extract_non_ticket_results(tool_results)
                    parts: list[str] = []

                    if other_results:
                        result_str = json.dumps(other_results, ensure_ascii=False, indent=2)
                        if len(result_str) >= 10:
                            try:
                                other_reply = await _generate(llm, [
                                    SystemMessage(content=get_prompt("response", RESPONSE_PROMPT).replace(
                                        "{intent}", f"{intent}查询"
                                    ).replace("{mcp_result}", result_str[:2000])),
                                    HumanMessage(content="请根据以上信息生成友好的中文回复"),
                                ], stream=not needs_review)
                                if other_reply and len(other_reply.strip()) > 5:
                                    parts.append(other_reply)
                            except Exception as e:
                                logger.error(f"LLM 格式化失败: {type(e).__name__}: {e}")

                    parts.append(ticket_table)
                    # 车票表格逐行流式输出；
                    # 流式接口下若只拼进 final_response，divs 非空时 __final__ 兜底被忽略，表格会丢失
                    if not needs_review:
                        from .metrics import get_stream_queue
                        q = get_stream_queue()
                        if q is not None:
                            await q.put(("generate_response", "\n\n"))
                            lines = ticket_table.splitlines(keepends=True)
                            for i, line in enumerate(lines):
                                await q.put(("generate_response", line))
                                if i < len(lines) - 1:
                                    await asyncio.sleep(0.02)
                    try:
                        suggestion = await _generate(llm, [
                            SystemMessage(content="你是车票助手。基于下面的车票表格，用一两句口语建议告诉用户该选哪趟（最快、卧铺、余票等）。只针对车票给建议，不要提及天气或其他需求。不要重新生成表格、不要省略任何车次。"),
                            HumanMessage(content=f"车票表格：\n{ticket_table}"),
                        ], stream=not needs_review)
                        if suggestion and len(suggestion.strip()) > 3:
                            parts.append(f"💡 {suggestion.strip()}")
                    except Exception as e:
                        logger.error(f"车票建议生成失败: {type(e).__name__}: {e}")

                    final_response = "\n\n".join(parts)
                else:
                    # 无车票表格（或无车票）：原有 LLM 汇总逻辑
                    result_str = json.dumps(tool_results, ensure_ascii=False, indent=2)
                    if len(result_str) < 10:
                        final_response = "😅 抱歉，没有查询到相关结果。"
                    else:
                        try:
                            content = await _generate(llm, [
                                SystemMessage(content=get_prompt("response", RESPONSE_PROMPT).replace(
                                    "{intent}", f"{intent}查询"
                                ).replace("{mcp_result}", result_str[:2000])),
                                HumanMessage(content="请根据以上信息生成友好的中文回复"),
                            ], stream=not needs_review)
                            if content and len(content.strip()) > 5:
                                final_response = content
                            else:
                                final_response = format_raw_result(intent, tool_results[0].get("result", {}))
                        except Exception as e:
                            logger.error(f"LLM 格式化失败: {type(e).__name__}: {e}")
                            final_response = format_raw_result(intent, tool_results[0].get("result", {}))

    if final_response is None:
        final_response = "😊 有什么我可以帮你的吗？"

    # 记忆固化：从本轮对话提取值得长期记住的事实（失败不影响主流程）
    if memory and user_text:
        asyncio.create_task(_async_remember(memory, llm, user_text, final_response))

    return {"final_response": final_response, "memory_context": memory_context}