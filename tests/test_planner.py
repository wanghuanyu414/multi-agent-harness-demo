import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from agent.harness.planner import planner_node


class FakeLLM:
    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=(
            '{"reasoning": "先查天气", "steps": [{"step": 0, "tool": "get_weather", "reason": "查天气"}]}'
        ))


def _state():
    return {
        "intent": "weather",
        "tool_calls": [{"name": "get_weather", "args": {"city": "北京"}, "id": "c1"}],
        "messages": [HumanMessage(content="北京天气")],
    }


def test_planner_keeps_tool_calls():
    """planner 不覆盖 function calling 的 tool_calls（消除与 analyze_intent 的重叠）"""
    state = _state()
    result = asyncio.run(planner_node(state, FakeLLM()))
    assert result["tool_calls"] == state["tool_calls"]
    assert result["plan_reasoning"] == "先查天气"
    assert result["execution_plan"][0]["tool"] == "get_weather"


def test_planner_chat_noop():
    result = asyncio.run(planner_node({"intent": "chat", "tool_calls": []}, FakeLLM()))
    assert result == {"intent": "chat"}
