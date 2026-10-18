"""Skill 注册表 —— 可插拔的领域能力单元（SKILL.md 文件驱动）

一个 Skill 封装：名字、展示标签、意图、职责描述、system prompt、绑定的工具。
Supervisor 路由、agent 构建、工具绑定全部从 SKILLS 列表生成。
新增技能 = 在项目根 skills/ 目录加一个文件夹 + SKILL.md（frontmatter + 正文=prompt），不改代码。
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain.agents import create_agent

from ..date_utils import parse_date
from ..langsmith_client import get_prompt
from ..metrics import ModelTimingMiddleware
from .skill_loader import load_skill_files, load_skill_prompt

logger = logging.getLogger(__name__)

# 项目根 skills/ 目录（agentskills.io 规范位置）
SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


@dataclass
class Skill:
    name: str            # worker 节点名 / agent 名，如 "weather_agent"
    label: str           # 展示标签，如 "🌤️ 天气"
    intent: str          # 意图标签，如 "weather"
    description: str     # 职责描述（动态生成 supervisor 路由 prompt 用）
    tools: list[str] = field(default_factory=list)    # 显式点名要用的 MCP 工具名（优先）
    domains: list[str] = field(default_factory=list)  # 兜底：按域继承该域全部工具
    local_tools: list = field(default_factory=list)   # 本地工具对象（如 parse-date），直接挂载
    _skill_md_path: Path | None = field(default=None, repr=False)  # SKILL.md 路径（第 2 级按需加载用）
    _prompt_cache: str | None = field(default=None, repr=False)    # 正文缓存

    @property
    def prompt(self) -> str:
        """worker 的 system prompt（渐进式披露第 2 级）：首次访问才读 SKILL.md 正文，之后缓存。"""
        if self._prompt_cache is None:
            if self._skill_md_path is None:
                return ""
            self._prompt_cache = load_skill_prompt(self._skill_md_path)
        return self._prompt_cache

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


# 本地工具注册表：本地工具是代码实现的，SKILL.md 通过 local_tools 字段按名字引用
LOCAL_TOOLS: dict[str, StructuredTool] = {
    "parse-date": PARSE_DATE_TOOL,
}


def _build_skill(raw: dict) -> Skill:
    """把 SKILL.md 解析出的 raw dict 构造成 Skill 对象"""
    local_names = raw.get("local_tools") or []
    return Skill(
        name=raw["name"],
        label=raw.get("label") or raw["name"],
        intent=raw.get("intent") or "chat",
        description=raw.get("description") or "",
        tools=list(raw.get("tools") or []),
        domains=list(raw.get("domains") or []),
        local_tools=[LOCAL_TOOLS[n] for n in local_names if n in LOCAL_TOOLS],
        _skill_md_path=raw.get("_skill_md_path"),
    )


SKILLS: list[Skill] = [_build_skill(raw) for raw in load_skill_files(SKILLS_DIR)]


def build_skill_agent(llm, skill: Skill, tools: list):
    """把一个 Skill 构建成 ReAct agent"""
    return create_agent(
        llm,
        tools,
        system_prompt=get_prompt(skill.intent, skill.prompt),
        name=skill.name,
        middleware=[ModelTimingMiddleware(skill.name)],
    )
