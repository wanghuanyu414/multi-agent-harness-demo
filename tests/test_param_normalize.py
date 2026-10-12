import asyncio
from datetime import date, timedelta

from agent.harness.param_normalize import (
    build_param_normalize_graph,
    parse_date_node,
    normalize_city_node,
    validate_required_node,
    build_tool_args_node,
)
from agent.builtin_tools import normalize_city_name


def _run(state: dict, llm=None) -> dict:
    return asyncio.run(build_param_normalize_graph(llm).ainvoke(state))


class _FakeMsg:
    def __init__(self, tool_calls):
        self.content = ""
        self.tool_calls = tool_calls


class FakeToolLLM:
    """模拟 bind_tools + tool_choice 的 LLM：按系统提示返回对应工具调用"""
    def bind_tools(self, tools, tool_choice=None):
        return self

    async def ainvoke(self, messages, **kwargs):
        sys = messages[0].content
        if "YYYY-MM-DD" in sys or "parse-date" in sys:
            return _FakeMsg([{"name": "parse-date", "args": {"date": "2026-10-01"}, "id": "c1"}])
        if "标准城市名" in sys:
            return _FakeMsg([{"name": "resolve-city", "args": {"city": "北京"}, "id": "c1"}])
        return _FakeMsg([])


# ============================================================
# 子图集成（L0）
# ============================================================

def test_subgraph_weather_normalize():
    out = _run({"intent": "weather", "extracted_params": {"city": "beijing"}, "user_text": "北京天气"})
    assert out["error"] == ""
    assert out["tool_args"] == {"city": "北京"}
    assert out["normalized_params"]["city"] == "北京"


def test_subgraph_weather_missing_city():
    out = _run({"intent": "weather", "extracted_params": {}, "user_text": "天气"})
    assert "城市" in out["error"]
    assert out["tool_args"] == {}


def test_subgraph_ticket_full():
    out = _run({
        "intent": "ticket",
        "extracted_params": {"from_city": "上海", "to_city": "北京", "date": "明天"},
        "user_text": "明天从上海到北京的高铁",
    })
    assert out["error"] == ""
    assert out["tool_args"]["from_station"] == "上海"
    assert out["tool_args"]["to_station"] == "北京"
    assert out["tool_args"]["train_date"] == (date.today() + timedelta(days=1)).isoformat()


def test_subgraph_ticket_missing_date():
    out = _run({
        "intent": "ticket",
        "extracted_params": {"from_city": "上海", "to_city": "北京", "date": None},
        "user_text": "上海到北京的票",
    })
    assert "出发日期" in out["error"]
    assert out["tool_args"] == {}


def test_subgraph_chat_passthrough():
    out = _run({"intent": "chat", "extracted_params": {}, "user_text": "你好"})
    assert out["error"] == ""
    assert out["tool_args"] == {}


# ============================================================
# L2（模糊日期 / 城市别名，走 LLM + 强制工具）
# ============================================================

def test_l2_resolve_fuzzy_date():
    out = _run({
        "intent": "ticket",
        "extracted_params": {"from_city": "上海", "to_city": "北京", "date": "下周三"},
        "user_text": "下周三从上海到北京",
    }, llm=FakeToolLLM())
    # L0 正则解析不了「下周三」→ 走 L2，FakeToolLLM 返回 2026-10-01
    assert out["date_parsed"] == "2026-10-01"
    assert out["error"] == ""


def test_l2_resolve_city_alias():
    out = _run({
        "intent": "weather",
        "extracted_params": {"city": "帝都"},
        "user_text": "帝都天气",
    }, llm=FakeToolLLM())
    # L0 字典不认识「帝都」→ 走 L2，FakeToolLLM 返回「北京」
    assert out["normalized_params"]["city"] == "北京"
    assert out["tool_args"] == {"city": "北京"}


# ============================================================
# 节点级
# ============================================================

async def test_parse_date_ticket_relative():
    r = await parse_date_node({"intent": "ticket", "extracted_params": {"date": "后天"}, "user_text": ""})
    assert r["date_parsed"] == (date.today() + timedelta(days=2)).isoformat()


async def test_parse_date_non_ticket_noop():
    r = await parse_date_node({"intent": "weather", "extracted_params": {}, "user_text": ""})
    assert r["date_parsed"] == ""


async def test_normalize_city_english_and_suffix():
    r = await normalize_city_node({"intent": "weather", "extracted_params": {"city": "beijing市"}, "user_text": ""})
    assert r["normalized_params"]["city"] == "北京"


async def test_normalize_city_ticket():
    r = await normalize_city_node({
        "intent": "ticket",
        "extracted_params": {"from_city": "shanghai", "to_city": "北京"},
        "user_text": "",
    })
    assert r["normalized_params"]["from_city"] == "上海"
    assert r["normalized_params"]["to_city"] == "北京"


def test_validate_weather_ok():
    r = validate_required_node({"intent": "weather", "normalized_params": {"city": "北京"}, "date_parsed": ""})
    assert r["error"] == ""


def test_build_tool_args_ticket():
    r = build_tool_args_node({
        "intent": "ticket", "error": "",
        "normalized_params": {"from_city": "上海", "to_city": "北京"},
        "date_parsed": "2024-01-01",
    })
    assert r["tool_args"] == {"train_date": "2024-01-01", "from_station": "上海", "to_station": "北京"}


def test_normalize_city_name():
    assert normalize_city_name("beijing") == "北京"
    assert normalize_city_name("北京市") == "北京"
    assert normalize_city_name("") == ""
    assert normalize_city_name(None) == ""
