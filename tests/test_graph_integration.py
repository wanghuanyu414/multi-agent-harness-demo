"""单智能体图端到端集成测试（Fake LLM + Fake MCP，无网络无 key）。

验证 normalize_params 子图在父图中的接线：意图 → 子图规范化 → 工具参数 → 回复。
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph


class FakeLLM:
    async def ainvoke(self, messages, **kwargs):
        sys = messages[0].content if messages else ""
        if "意图分析器" in sys:
            return AIMessage(
                content='{"intent":"weather","city":"beijing","from_city":null,"to_city":null,"date":null}'
            )
        return AIMessage(content="北京今天晴，20℃，适合出行。")


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
        return []

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


def test_single_agent_weather_normalization():
    app = build_graph(FakeLLM(), FakeMCP(), config=_config())

    async def run():
        return await app.ainvoke(
            {"messages": [HumanMessage(content="北京天气怎么样")]},
            {"configurable": {"thread_id": "test-weather"}},
        )

    result = asyncio.run(run())
    # 子图把英文 city "beijing" 规范化为 "北京"
    assert result.get("normalized_params", {}).get("city") == "北京"
    assert result.get("tool_args") == {"city": "北京"}
    assert "晴" in result.get("final_response", "")


def test_single_agent_ticket_missing_date_routes_to_error():
    class TicketLLM(FakeLLM):
        async def ainvoke(self, messages, **kwargs):
            sys = messages[0].content if messages else ""
            if "意图分析器" in sys:
                return AIMessage(
                    content='{"intent":"ticket","city":null,"from_city":"上海","to_city":"北京","date":null}'
                )
            return AIMessage(content="")

    app = build_graph(TicketLLM(), FakeMCP(), config=_config())

    async def run():
        return await app.ainvoke(
            {"messages": [HumanMessage(content="上海到北京的票")]},
            {"configurable": {"thread_id": "test-ticket"}},
        )

    result = asyncio.run(run())
    # 缺日期 → 子图设 error → 主图 tool_router 路由到 generate_response 返回错误提示
    assert "出发日期" in result.get("final_response", "")
