"""LangGraph 工作流状态定义"""
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Agent 工作流状态

    Attributes:
        messages: 对话消息历史
        intent: 识别的用户意图 (weather / ticket / chat)
        extracted_params: 从用户消息中提取的参数
        mcp_server: 目标 MCP 服务器名称
        tool_name: 要调用的工具名称
        tool_args: 工具调用参数
        mcp_result: MCP 工具调用返回的原始结果
        final_response: 最终生成的回复文本
        error: 错误信息（如有）
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str
    extracted_params: dict
    user_text: str                 # 最新用户消息原文（参数规范化子图日期兜底用）
    normalized_params: dict        # 参数规范化子图产出的干净参数
    date_parsed: str               # 解析后的出发日期 YYYY-MM-DD
    mcp_server: str
    tool_name: str
    tool_args: dict
    mcp_result: dict
    final_response: str
    error: str

    # === Harness: Guardrails ===
    input_validated: bool
    input_guardrail_result: dict    # {"passed": bool, "violations": [...], "mode": "blocking|warning"}
    output_guardrail_result: dict   # {"passed": bool, "violations": [...]}

    # === Harness: Reflection Loop ===
    retry_count: int
    max_retries: int
    reflection_notes: str           # "retry: <reason>" or "ok: <reason>"

    # === Harness: Dynamic Planning ===
    execution_plan: list            # [{"step": 0, "tool_server": "...", "tool_name": "...", "tool_args": {...}, "reason": "..."}]
    current_step: int
    plan_reasoning: str             # 规划的整体 CoT 推理
    step_results: list              # 每步执行结果累积 [{"step": 0, "tool_name": "...", "result": {...}}]

    # === Harness: Observability ===
    trace_id: str
    span_chain: list                # [{"node": "...", "duration_ms": 123, ...}]

    # === 上下文治理 + 长期记忆 + 人工审核（单/多 Agent 共用） ===
    conversation_summary: str       # 长会话压缩摘要（summarize 节点写入）
    memory_context: str             # 本轮召回的长期记忆上下文
    human_review: dict              # 人工审核结果 {"status": approved/rejected/skipped, "intent": ...}