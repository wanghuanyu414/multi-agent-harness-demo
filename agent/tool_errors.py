"""工具调用统一错误模型 —— 生产级失败分类与用户友好文案

把底层异常归类为 ToolCallError，供上层做差异化兜底：
- code: 错误码，用于指标分桶与告警
- retryable: 是否值得重试（决定 mcp_client 是否退避重试）
- user_message: 面向用户的友好文案，可直接展示
- detail: 内部细节，只进日志不进回复
"""
from __future__ import annotations

import asyncio

# 错误码常量
TIMEOUT = "TIMEOUT"
CONNECTION = "CONNECTION"
SERVER_ERROR = "SERVER_ERROR"
NOT_FOUND = "NOT_FOUND"
BAD_ARGS = "BAD_ARGS"
UNKNOWN = "UNKNOWN"


class ToolCallError(Exception):
    """工具调用失败的统一异常，携带分类信息。"""

    def __init__(self, code: str, user_message: str, *, retryable: bool = False, detail: str = ""):
        self.code = code
        self.user_message = user_message
        self.retryable = retryable
        self.detail = detail
        super().__init__(user_message or code)


def classify_exception(exc: Exception) -> ToolCallError:
    """把底层异常归类为 ToolCallError。

    只有可重试的异常（超时/连接/服务端错误）才值得重试；
    参数错误、资源不存在这类确定性失败直接返回，重试无意义。
    """
    if isinstance(exc, ToolCallError):
        return exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ToolCallError(TIMEOUT, "服务响应超时，请稍后重试", retryable=True, detail=str(exc))
    if isinstance(exc, ConnectionError):
        return ToolCallError(CONNECTION, "服务暂时不可用，请稍后重试", retryable=True, detail=str(exc))
    if isinstance(exc, RuntimeError):
        # JSON-RPC 错误等多属服务端错误，可重试
        return ToolCallError(SERVER_ERROR, "服务内部错误，请稍后重试", retryable=True, detail=str(exc))
    if isinstance(exc, ValueError):
        # 参数不合法等确定性错误
        return ToolCallError(BAD_ARGS, "请求参数有误，请检查后重试", retryable=False, detail=str(exc))
    return ToolCallError(UNKNOWN, "请求处理失败，请稍后重试", retryable=False, detail=str(exc))