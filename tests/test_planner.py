from agent.harness.planner import _validate_plan


class FakeMCP:
    def __init__(self, tools):
        self._tools = tools

    def find_tool(self, server, hint=""):
        for srv, name in self._tools:
            if srv == server and (not hint or hint.lower() in name.lower()):
                return {"name": name}
        return None


def test_validate_plan_valid():
    mcp = FakeMCP([("weather", "get_weather")])
    plan = {"steps": [{"tool_server": "weather", "tool_name": "get_weather"}]}
    assert _validate_plan(plan, mcp) is True


def test_validate_plan_missing_tool():
    mcp = FakeMCP([("weather", "get_weather")])
    plan = {"steps": [{"tool_server": "weather", "tool_name": "nope"}]}
    assert _validate_plan(plan, mcp) is False


def test_validate_plan_missing_fields():
    mcp = FakeMCP([("weather", "get_weather")])
    plan = {"steps": [{"tool_name": "get_weather"}]}  # 缺 tool_server
    assert _validate_plan(plan, mcp) is False


def test_validate_plan_empty():
    mcp = FakeMCP([])
    assert _validate_plan(None, mcp) is False
    assert _validate_plan({}, mcp) is False
    assert _validate_plan({"steps": []}, mcp) is False
