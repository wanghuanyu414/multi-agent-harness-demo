"""内置工具 —— 不依赖外部 MCP 服务器的工具实现"""
import json
import logging
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


def weather_builtin(city: str) -> dict:
    """通过中国天气网（d1.weather.com.cn）查询城市实时天气（无需 API Key）

    Args:
        city: 城市名称（中文或英文）

    Returns:
        结构化天气数据（温度、天气状况、湿度、风向风力、空气质量等）
    """
    code = _resolve_city_code(city) or _search_city_code(city)
    if not code:
        return {"error": f"未找到城市「{city}」的天气信息，请检查城市名称是否正确"}

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
        "description": "查询城市实时天气信息（通过中国天气网）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如 北京、上海"}
            },
            "required": ["city"],
        },
        "handler": weather_builtin,
    }
}
