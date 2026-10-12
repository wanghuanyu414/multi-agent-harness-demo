"""Supervisor + Worker 多 Agent 共享状态定义"""
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MultiAgentState(TypedDict):
    """多 Agent 系统共享状态

    所有 Agent（Supervisor + 专职 Worker）通过同一份状态协同：
    - messages: 共享对话历史（Worker 中间 ReAct 轨迹不回写，只回写最终 AIMessage）
    - next: Supervisor 决定的下一个节点
    - intent: 识别的用户意图（weather / ticket / chat）
    - final_response: Worker 产出的最终回复
    - agent_trace: 本次请求经过的 Agent 名称序列（可观测性）
    - conversation_summary: 长会话压缩摘要（超阈值时把旧消息压成要点，跨轮保留）
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    intent: str
    final_response: str
    agent_trace: list  # 本次请求的 Agent 链路（每轮由 supervisor 重置）
    memory_context: str  # 本轮召回的长期记忆上下文（注入给专职 Worker）
    conversation_summary: str  # 压缩后的历史对话摘要（注入 Supervisor 路由与 Worker）
    pending_agents: list  # 待派发的 Worker 队列（多步路由）
    worker_results: list  # 各 Worker 回复累积 [{"agent": ..., "response": ...}]
    tool_results: list  # 各 Worker 实际调用的工具原始结果累积（供输出护栏事实性校验）
    plan_reasoning: str  # Supervisor 的派发推理（CoT）
    human_review: dict  # 人工审核结果 {"status": approved/rejected/skipped, "intent": ...}
    input_guardrail_result: dict  # 输入护栏结果 {"passed": bool, "violations": [...], "mode": ...}
    output_guardrail_result: dict  # 输出护栏结果 {"passed": bool, "violations": [...]}
