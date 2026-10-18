"""单智能体图端到端集成测试（Fake LLM + Fake MCP，无网络无 key）。

验证 function calling 流程。
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph


class FakeLLM:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        # 模拟 function calling：返回天气工具调用
        return AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"city": "北京"}, "id": "call1"}],
        )


class FakeMCP:
    def is_server_running(self, name):
        return True

    def get_startup_error(self, name):
        return None

    def find_tool(self, server, hint=""):
        return {"name": "get_weather"}

    def get_all_tools_description(self):
        return "get_weather: 查询天气"

    def list_tools(self):
        return [
            {"name": "get_weather", "server": "builtin", "description": "查询天气",
             "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
            {"name": "query-tickets", "server": "12306", "description": "查询车票",
             "inputSchema": {"type": "object", "properties": {"from_station": {"type": "string"}, "to_station": {"type": "string"}, "date": {"type": "string"}}, "required": ["from_station", "to_station", "date"]}},
        ]

    def unhealthy_servers(self):
        return []

    async def call_tool(self, server, tool, args):
        return {"content": [{"type": "text", "text": f"北京 晴 20℃ (city={args.get('city')})"}]}


def _config():
    return {
        "harness": {
            "guardrails": {"enabled": False},
            "planner": {"enabled": False},
            "reflection": {"enabled": False},
            "observability": {"enabled": False},
            "human_review": {"enabled": False},
        },
        "context": {"summarize": {"enabled": False}},
    }


def test_single_agent_weather_function_calling():
    app = build_graph(FakeLLM(), FakeMCP(), config=_config())

    async def run():
        return await app.ainvoke(
            {"messages": [HumanMessage(content="北京天气怎么样")]},
            {"configurable": {"thread_id": "test-weather"}},
        )

    result = asyncio.run(run())
    # function calling 产出工具调用，并行执行后汇总
    assert result.get("tool_calls"), "应有工具调用"
    assert result.get("tool_results"), "应有工具结果"
    assert "晴" in result.get("final_response", "")


def test_single_agent_ticket_function_calling():
    class TicketLLM(FakeLLM):
        async def ainvoke(self, messages, **kwargs):
            return AIMessage(
                content="",
                tool_calls=[{"name": "query-tickets", "args": {"from_station": "上海", "to_station": "北京", "date": "2026-10-17"}, "id": "call1"}],
            )

    app = build_graph(TicketLLM(), FakeMCP(), config=_config())

    async def run():
        return await app.ainvoke(
            {"messages": [HumanMessage(content="上海到北京的票")]},
            {"configurable": {"thread_id": "test-ticket"}},
        )

    result = asyncio.run(run())
    assert result.get("tool_calls"), "应有工具调用"
    assert result.get("tool_results"), "应有工具结果"
    assert "晴" in result.get("final_response", "")
