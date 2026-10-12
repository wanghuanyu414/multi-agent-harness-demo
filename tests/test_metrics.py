from agent.metrics import count_tokens, percentiles, MetricsCollector


def test_count_tokens_empty():
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_count_tokens_positive():
    assert count_tokens("hello world") > 0
    assert count_tokens("北京天气") > 0


def test_percentiles_empty():
    assert percentiles([]) == {"p50": None, "p95": None, "p99": None}


def test_percentiles():
    r = percentiles([1, 2, 3, 4, 5])
    assert r["p50"] == 3.0
    assert r["p99"] is not None


def test_collector_record_and_summary():
    c = MetricsCollector.get()
    c.reset()
    c.record_tool("get_weather", True, 10.0, tokens=5)
    c.record_tool("get_weather", False, 20.0)
    s = c.tool_summary()
    assert s["get_weather"]["calls"] == 2
    assert s["get_weather"]["success_rate"] == 0.5
    c.reset()


def test_collector_tokens():
    c = MetricsCollector.get()
    c.reset()
    c.add_tokens(input_tokens=10, output_tokens=5)
    assert c.token_summary()["total"] == 15
    c.reset()


def test_collector_model_latency():
    c = MetricsCollector.get()
    c.reset()
    c.record_model("weather_agent", 5.0)
    c.record_model("ticket_agent", 10.0)
    s = c.model_latency_summary()
    assert set(s.keys()) == {"weather_agent", "ticket_agent"}
    assert s["weather_agent"]["p50"] == 5.0
    c.reset()
