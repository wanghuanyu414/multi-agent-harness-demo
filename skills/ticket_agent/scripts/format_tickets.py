#!/usr/bin/env python3
"""确定性格式化 12306 车票查询结果为 Markdown 表格。

标准 SKILL.md 第 3 级脚本：stdin 读 query-tickets 的 JSON 结果，
stdout 输出完整 Markdown 表格（遍历全部车次，不省略）。
只有输出进入 LLM 上下文，脚本代码本身不加载。

用法：python format_tickets.py < result.json
"""
import json
import sys

# 席别键 → 中文名（query-tickets 返回的 seats 字段键）
SEAT_NAMES = {
    "business": "商务",
    "first_class": "一等",
    "second_class": "二等",
    "soft_sleeper": "软卧",
    "hard_sleeper": "硬卧",
    "hard_seat": "硬座",
    "no_seat": "无座",
}


def _format_seats(seats: dict) -> str:
    """把 seats dict 格式化成「商务4 · 一等11 · 二等有」这样的余票串"""
    parts = []
    for key, label in SEAT_NAMES.items():
        val = seats.get(key)
        if val is None:
            continue
        val = str(val).strip()
        if val in ("", "无", "0"):
            continue
        # 数值型余票显示「张」，字符串型（"有"）直接显示
        parts.append(f"{label}{val}")
    return " · ".join(parts) if parts else "无票"


def _format_duration(duration: str) -> str:
    """把 HH:MM 历时格式化成「X小时Y分」"""
    try:
        h, m = duration.split(":")
        h, m = int(h), int(m)
        if h and m:
            return f"{h}小时{m}分"
        if h:
            return f"{h}小时"
        return f"{m}分"
    except (ValueError, AttributeError):
        return duration


def format_tickets(data: dict) -> str:
    """把 query-tickets 的 JSON 结果格式化成 Markdown 表格"""
    if not isinstance(data, dict):
        return "车票查询结果解析失败"

    if not data.get("success", True):
        return f"车票查询失败：{data.get('message', '未知错误')}"

    trains = data.get("trains", [])
    if not trains:
        return "未查询到符合条件的车票，请检查出发地/目的地/日期后重试。"

    from_station = data.get("from_station", "")
    to_station = data.get("to_station", "")
    train_date = data.get("train_date", "")
    header = f"🚄 **{from_station} → {to_station}**（{train_date}，共 {len(trains)} 趟）\n\n"

    lines = ["| 车次 | 出发 → 到达 | 时间 | 历时 | 余票 |",
             "|------|------------|------|------|------|"]
    for t in trains:
        train_no = t.get("train_no", "")
        f_station = t.get("from_station", "")
        t_station = t.get("to_station", "")
        start = t.get("start_time", "")
        arrive = t.get("arrive_time", "")
        duration = _format_duration(t.get("duration", ""))
        seats = _format_seats(t.get("seats", {}))
        lines.append(
            f"| {train_no} | {f_station} → {t_station} | "
            f"{start} → {arrive} | {duration} | {seats} |"
        )

    return header + "\n".join(lines)


def main() -> None:
    # Windows 下 stdin/stdout 默认 GBK，无法输出 emoji/中文；强制 UTF-8（脚本约定 UTF-8 进出）
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"输入不是合法 JSON：{e}", file=sys.stderr)
        sys.exit(1)
    print(format_tickets(data))


if __name__ == "__main__":
    main()
