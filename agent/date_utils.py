"""日期解析工具 —— 把中文/相对日期解析为 YYYY-MM-DD 格式（单 agent 与多 agent 共用）"""
import re
from datetime import date, timedelta

_CHINESE_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
                "两": 2}

# 相对日期词
_RELATIVE_DATE: dict[str, int] = {
    "今天": 0, "今日": 0,
    "明天": 1, "明日": 1,
    "后天": 2, "后日": 2,
    "大后天": 3,
    "昨天": -1, "昨日": -1,
    "前天": -2, "前日": -2,
}


def parse_date(text: str) -> str | None:
    """解析中文日期为 YYYY-MM-DD 格式

    支持格式:
        - 今天/明天/后天/昨天/前天
        - 2024-01-01, 2024/01/01
        - 1月1日, 2024年1月1日
        - 1月1号, 1.1

    解析失败返回 None（由调用方提示用户，而非静默当作今天）。
    """
    if not text:
        return None
    today = date.today()

    # 1) 相对日期
    if text in _RELATIVE_DATE:
        return (today + timedelta(days=_RELATIVE_DATE[text])).isoformat()

    # 2) ISO 格式 YYYY-MM-DD / YYYY/MM/DD
    m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        return f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"

    # 3) "2024年1月1日" / "2024年1月1号"
    m = re.match(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?", text)
    if m:
        return f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"

    # 4) "1月1日" / "1月1号" — 缺少年份，用今年
    m = re.match(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?", text)
    if m:
        return f"{today.year:04d}-{int(m[1]):02d}-{int(m[2]):02d}"

    # 5) 中文数字 "五月一日" 等
    m = re.match(r"([一二三四五六七八九十]{1,3})\s*月\s*([一二三四五六七八九十廿]{1,3})\s*[日号]?", text)
    if m:
        month = _CHINESE_NUM.get(m[1], 0)
        day = _CHINESE_NUM.get(m[2], 0)
        if month and day:
            return f"{today.year:04d}-{month:02d}-{day:02d}"

    # 6) "1.1" / "01.01"
    m = re.match(r"(\d{1,2})\.(\d{1,2})", text)
    if m:
        return f"{today.year:04d}-{int(m[1]):02d}-{int(m[2]):02d}"

    # 解析失败：返回 None，由调用方决定如何提示用户
    return None


def extract_date_from_message(text: str) -> str | None:
    """从用户消息中提取日期字符串"""
    # 相对日期
    for word in _RELATIVE_DATE:
        if word in text:
            return parse_date(word)

    # 绝对日期模式
    patterns = [
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*[日号]?",
        r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]?",
        r"[一二三四五六七八九十]{1,3}\s*月\s*[一二三四五六七八九十廿]{1,3}\s*[日号]?",
        r"\d{1,2}\.\d{1,2}",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return parse_date(m[0])

    return None
