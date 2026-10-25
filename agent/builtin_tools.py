"""内置工具 —— 不依赖外部 MCP 服务器的工具实现"""
import json
import logging
import re
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

# ============================================================
# 城市代码映射（中国天气网 d1.weather.com.cn）
# ============================================================

_CITY_CODE: dict[str, str] = {
    "北京": "101010100", "上海": "101020100", "天津": "101030100",
    "重庆": "101040100", "广州": "101280101", "深圳": "101280601",
    "杭州": "101210101", "南京": "101190101", "成都": "101270101",
    "武汉": "101200101", "西安": "101110101", "郑州": "101180101",
    "苏州": "101190401", "长沙": "101250101", "沈阳": "101070101",
    "青岛": "101120201", "合肥": "101220101", "佛山": "101280800",
    "大连": "101070201", "福州": "101230101", "厦门": "101230201",
    "哈尔滨": "101050101", "济南": "101120101", "温州": "101210701",
    "南宁": "101300101", "长春": "101060101", "泉州": "101230501",
    "贵阳": "101260101", "南昌": "101240101", "金华": "101210901",
    "常州": "101191101", "惠州": "101280301", "嘉兴": "101210301",
    "徐州": "101190801", "南通": "101190501", "太原": "101100101",
    "珠海": "101280701", "中山": "101281701", "保定": "101090201",
    "兰州": "101160101", "台州": "101210601", "绍兴": "101210501",
    "烟台": "101120501", "廊坊": "101090601", "乌鲁木齐": "101130101",
    "昆明": "101290101", "汕头": "101280501", "洛阳": "101180901",
    "呼和浩特": "101080101", "拉萨": "101140101", "银川": "101170101",
    "海口": "101310101", "三亚": "101310201", "桂林": "101300501",
    "丽江": "101291401", "张家界": "101251101", "黄山": "101221001",
}

# 已知城市集合（L2 城市归一化用：约束 LLM 只能输出这些标准城市名）
KNOWN_CITIES: frozenset = frozenset(_CITY_CODE.keys())

# 常见英文名 → 中文名（保持与原 wttr.in 版本兼容）
_EN_TO_CN: dict[str, str] = {
    "beijing": "北京", "shanghai": "上海", "guangzhou": "广州",
    "shenzhen": "深圳", "hangzhou": "杭州", "nanjing": "南京",
    "chengdu": "成都", "chongqing": "重庆", "wuhan": "武汉",
    "xian": "西安", "xi'an": "西安", "zhengzhou": "郑州",
    "tianjin": "天津", "changsha": "长沙", "qingdao": "青岛",
    "xiamen": "厦门", "dalian": "大连", "kunming": "昆明",
    "sanya": "三亚", "guilin": "桂林",
}


def normalize_city_name(city: str) -> str:
    """城市名归一化：去「市/县」后缀 → 英文→中文。用于参数规范化子图。"""
    if not city:
        return ""
    name = city.strip().rstrip("市县")
    if name.lower() in _EN_TO_CN:
        name = _EN_TO_CN[name.lower()]
    return name


def _resolve_city_code(city: str) -> str | None:
    """把城市名（中文或英文）解析成中国天气网城市代码"""
    if not city:
        return None
    return _CITY_CODE.get(normalize_city_name(city))


def _no_proxy_open(url: str, headers: dict, timeout: int = 10):
    """用无代理 opener 请求，绕过 Windows 系统代理"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers=headers)
    return opener.open(req, timeout=timeout)


def _search_city_code(city: str) -> str | None:
    """通过中国天气网搜索接口动态查询城市代码（兜底，支持任意城市）"""
    url = "http://toy1.weather.com.cn/search?cityname=" + urllib.parse.quote(city)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "http://www.weather.com.cn/",
    }
    try:
        with _no_proxy_open(url, headers) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # 响应是 JSONP 风格：([{...}])，提取 [...] 部分再解析
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            return None
        data = json.loads(raw[start:end + 1])
        if isinstance(data, list) and data and isinstance(data[0], dict):
            ref = data[0].get("ref", "")
            code = ref.split("~")[0] if "~" in ref else None
            if code and code.isdigit():
                return code
    except Exception as e:
        logger.warning(f"城市代码搜索失败: {e}")
    return None


def _fetch_7d_forecast(code: str) -> list[dict]:
    """获取 7 天预报（解析 HTML 页面 weather/{code}.shtml）"""
    url = f"http://www.weather.com.cn/weather/{code}.shtml"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "http://www.weather.com.cn/",
    }
    try:
        with _no_proxy_open(url, headers) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"7 天预报获取失败: {e}")
        return []

    m = re.search(r'<ul class="t clearfix">(.*?)</ul>', raw, re.S)
    if not m:
        return []

    days = []
    for li in re.findall(r'<li.*?</li>', m.group(1), re.S):
        h1 = re.search(r'<h1>(.*?)</h1>', li)
        wea = re.search(r'class="wea"[^>]*>(.*?)</p>', li)
        tem = re.search(r'<span>(.*?)</span>/<i>(.*?)</i>', li)
        win = re.search(r'class="win"[^>]*>(.*?)</p>', li, re.S)
        # 风向在 span 的 title 属性，等级在 <i> 里
        wind = ""
        if win:
            win_dir = re.search(r'title="([^"]+)"', win.group(1))
            win_level = re.search(r'<i>(.*?)</i>', win.group(1), re.S)
            wind = " ".join(x for x in [
                win_dir.group(1) if win_dir else "",
                re.sub(r'<[^>]+>', '', win_level.group(1)).strip() if win_level else "",
            ] if x)
        days.append({
            "date": h1.group(1).strip() if h1 else "",
            "weather_desc": re.sub(r'<[^>]+>', '', wea.group(1)).strip() if wea else "",
            "high_temp": tem.group(1).strip() if tem else "",
            "low_temp": tem.group(2).strip().rstrip("℃") if tem else "",
            "wind": wind,
        })
    return days


def weather_builtin(city: str, day: int = 0) -> dict:
    """查询城市天气（中国天气网，无需 API Key）。

    day=0 返回实时天气（温度/天气/湿度/风/空气质量）；
    day=1~6 返回未来第 N 天的预报（最高/最低温/天气/风）。

    同步实现（urllib 阻塞式请求），由调用方放入线程池执行，避免阻塞事件循环。
    """
    code = _resolve_city_code(city) or _search_city_code(city)
    if not code:
        return {"error": f"未找到城市「{city}」的天气信息，请检查城市名称是否正确"}

    # 未来预报
    if day > 0:
        days = _fetch_7d_forecast(code)
        if not days:
            return {"error": "天气预报服务暂时不可用，请稍后重试"}
        if day >= len(days):
            return {"error": f"仅支持未来 7 天预报（含今天），超出可查询范围"}
        d = days[day]
        return {
            "city": city,
            "date": d.get("date", ""),
            "weather_desc": d.get("weather_desc", "N/A"),
            "high_temp": d.get("high_temp", "N/A"),
            "low_temp": d.get("low_temp", "N/A"),
            "wind": d.get("wind", "N/A"),
        }

    # 实时
    url = f"http://d1.weather.com.cn/sk_2d/{code}.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "http://www.weather.com.cn/",
    }

    try:
        with _no_proxy_open(url, headers) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        # 响应形如: var dataSK={...}
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return {"error": f"天气数据解析失败：{raw[:100]}"}
        data = json.loads(raw[start:end + 1])

        return {
            "city": data.get("cityname", city),
            "temperature": data.get("temp", "N/A"),
            "weather_desc": data.get("weather", "N/A"),
            "wind_dir": data.get("WD", "N/A"),
            "wind_speed": data.get("WS", "N/A"),
            "humidity": data.get("SD", "N/A"),
            "aqi": data.get("aqi", "N/A"),
            "observation_time": data.get("time", "N/A"),
            "date": data.get("date", "N/A"),
        }
    except urllib.error.URLError as e:
        logger.error(f"天气查询失败: {e}")
        return {"error": f"天气服务暂时不可用: {e.reason}"}
    except Exception as e:
        logger.error(f"天气解析失败: {e}")
        return {"error": f"天气数据解析失败: {str(e)}"}


BUILTIN_TOOLS: dict[str, dict] = {
    "get_weather": {
        "name": "get_weather",
        "description": "查询城市天气（今天实时或未来 7 天预报）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如 北京、上海"},
                "day": {"type": "integer", "description": "预报天数偏移：0=今天(实时)，1=明天，2=后天，最多 6"},
            },
            "required": ["city"],
        },
        "handler": weather_builtin,
    }
}
