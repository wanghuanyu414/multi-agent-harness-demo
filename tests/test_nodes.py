from agent.nodes import route_tool_node, format_raw_result, _keyword_intent


def test_route_weather():
    state = {"intent": "weather"}
    r = route_tool_node(state)
    assert r["mcp_server"] == "weather"
    assert r["tool_name"] == "get_weather"


def test_route_ticket():
    state = {"intent": "ticket"}
    r = route_tool_node(state)
    assert r["mcp_server"] == "12306"
    assert r["tool_name"] == "query-tickets"


def test_route_chat():
    state = {"intent": "chat"}
    r = route_tool_node(state)
    assert "mcp_server" not in r


def test_format_raw_result():
    r = format_raw_result("weather", {"content": [{"type": "text", "text": "晴 20℃"}]})
    assert "晴" in r


def test_keyword_intent_ticket():
    r = _keyword_intent("明天从北京到上海的高铁")
    assert r["intent"] == "ticket"
    assert r["from_city"] == "北京"
    assert r["to_city"] == "上海"


def test_keyword_intent_weather():
    r = _keyword_intent("北京天气怎么样")
    assert r["intent"] == "weather"
    assert r["city"] == "北京"
