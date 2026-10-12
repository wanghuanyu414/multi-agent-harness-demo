"""LLM 调用与解析通用工具 —— 指数退避重试 + JSON 容错解析

消除 nodes / guardrails / planner / supervisor / memory 里重复的
「代码围栏剥离 + json.loads」逻辑，统一为一个健壮的 parse_llm_json。
"""
import asyncio
import json
import logging
import re

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def parse_llm_json(content):
    """从 LLM 输出中容错解析 JSON。

    依次处理：markdown 代码围栏（```json ... ``` / 不成对的围栏）、
    前后空白、以及「首个 { 到最后一个 }（或 [ ... ]）」的兜底截取。

    Returns:
        解析出的对象（dict/list/标量），失败返回 None。
    """
    if content is None:
        return None
    text = str(content).strip()
    if not text:
        return None

    # 完整围栏
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    else:
        # 不成对的围栏：去掉开头 ```json 与结尾 ```
        if text.startswith("```"):
            rest = text[3:]
            text = rest.split("```", 1)[1] if "```" in rest else rest
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
            text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 兜底：截取第一个 { / [ 到最后一个 } / ]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


async def ainvoke_with_retry(llm, messages, retries: int = 2, backoff_base: float = 0.5):
    """带指数退避的 LLM 调用。

    最后一次失败时把异常抛出（由调用方按各自语义降级/兜底）。
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await llm.ainvoke(messages)
        except Exception as e:  # noqa: BLE001 - 各调用方有兜底，这里统一重试
            last_err = e
            if attempt < retries:
                wait = backoff_base * (2 ** attempt)
                logger.warning(
                    "LLM 调用失败(第 %d/%d 次)，%.1fs 后重试: %s",
                    attempt + 1, retries + 1, wait, type(e).__name__,
                )
                await asyncio.sleep(wait)
    raise last_err  # type: ignore[misc]
