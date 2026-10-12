from datetime import date, timedelta

from agent.date_utils import parse_date, extract_date_from_message


def test_parse_iso():
    assert parse_date("2024-01-05") == "2024-01-05"
    assert parse_date("2024/1/5") == "2024-01-05"


def test_parse_chinese_full():
    assert parse_date("2024年1月5日") == "2024-01-05"
    assert parse_date("2024年1月5号") == "2024-01-05"


def test_parse_month_day_current_year():
    today = date.today()
    assert parse_date("1月5日") == f"{today.year:04d}-01-05"


def test_parse_relative():
    today = date.today()
    assert parse_date("明天") == (today + timedelta(days=1)).isoformat()
    assert parse_date("后天") == (today + timedelta(days=2)).isoformat()
    assert parse_date("昨天") == (today - timedelta(days=1)).isoformat()


def test_parse_dot():
    today = date.today()
    assert parse_date("1.5") == f"{today.year:04d}-01-05"


def test_parse_invalid():
    assert parse_date("随便写写") is None
    assert parse_date(None) is None


def test_extract_date_relative():
    assert extract_date_from_message("明天从上海到北京") == (date.today() + timedelta(days=1)).isoformat()


def test_extract_date_absolute():
    assert extract_date_from_message("查2024-03-01的票") == "2024-03-01"


def test_extract_date_none():
    assert extract_date_from_message("你好") is None
