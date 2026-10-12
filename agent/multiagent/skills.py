"""Skill 注册表 —— 可插拔的领域能力单元

一个 Skill 封装：名字、展示标签、意图、职责描述、system prompt、绑定的工具。
Supervisor 路由、agent 构建、工具绑定全部从 SKILLS 列表生成。
新增技能 = 往 SKILLS 里加一个 Skill 对象。
"""
import logging
from dataclasses import dataclass, field

from langchain_core.tools import StructuredTool
from langchain.agents import create_agent

from ..date_utils import parse_date
from ..prompts import WEATHER_PROMPT, TICKET_PROMPT, CHAT_PROMPT
from ..langsmith_client import get_prompt
from ..metrics import ModelTimingMiddleware

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str            # worker 节点名 / agent 名，如 "weather_agent"
    label: str           # 展示标签，如 "🌤️ 天气"
    intent: str          # 意图标签，如 "weather"
    description: str     # 职责描述（动态生成 supervisor 路由 prompt 用）
    prompt: str          # worker 的 system prompt
    tools: list[str] = field(default_factory=list)    # 显式点名要用的 MCP 工具名（优先）
    domains: list[str] = field(default_factory=list)  # 兜底：按域继承该域全部工具
    local_tools: list = field(default_factory=list)   # 本地工具对象（如 parse-date），直接挂载

    def collect_tools(self, specialist_tools: dict) -> list:
        """收集本技能的工具：本地工具 + 显式 MCP 工具（或按域继承）"""
        result = list(self.local_tools)
        if self.tools:
            wanted = set(self.tools)
            found = []
            for domain_tools in specialist_tools.values():
                for t in domain_tools:
                    if getattr(t, "name", None) in wanted:
                        found.append(t)
            missing = wanted - {getattr(t, "name", None) for t in found}
            if missing:
                logger.warning(f"技能 {self.name} 声明了未注册的工具: {sorted(missing)}")
            result.extend(found)
        else:
            for d in self.domains:
                result.extend(specialist_tools.get(d, []))
        return result


def _parse_date_tool(date_text: str) -> str:
    """把相对日期或中文日期解析为 YYYY-MM-DD 格式"""
    parsed = parse_date(date_text)
    if not parsed:
        return f"无法解析日期「{date_text}」，请提供 YYYY-MM-DD 格式或明确的相对日期（今天/明天/后天）"
    return parsed


PARSE_DATE_TOOL = StructuredTool.from_function(
    func=_parse_date_tool,
    name="parse-date",
    description=(
        "把相对日期（今天/明天/后天/昨天）或中文日期（如 2024年1月1日、1月1日、5.1）"
        "解析为 YYYY-MM-DD 格式。查车票前如果用户给的日期不是具体 YYYY-MM-DD 日期，先用本工具换算。"
    ),
)


WEATHER_SKILL = Skill(
    name="weather_agent",
    label="🌤️ 天气",
    intent="weather",
    description="天气查询（用户问某地天气/气温/是否下雨等）",
    prompt=WEATHER_PROMPT,
    tools=["get_weather"],
)

TICKET_SKILL = Skill(
    name="ticket_agent",
    label="🚄 车票",
    intent="ticket",
    description="火车票/高铁票查询（用户问车次/订票/余票等）",
    prompt=TICKET_PROMPT,
    tools=["query-tickets", "search-stations"],
    local_tools=[PARSE_DATE_TOOL],
)

CHAT_SKILL = Skill(
    name="chat_agent",
    label="💬 对话",
    intent="chat",
    description="闲聊、问候、一般对话",
    prompt=CHAT_PROMPT,
    tools=[],
)

SKILLS: list[Skill] = [WEATHER_SKILL, TICKET_SKILL, CHAT_SKILL]


def build_skill_agent(llm, skill: Skill, tools: list):
    """把一个 Skill 构建成 ReAct agent"""
    return create_agent(
        llm,
        tools,
        system_prompt=get_prompt(skill.intent, skill.prompt),
        name=skill.name,
        middleware=[ModelTimingMiddleware(skill.name)],
    )
