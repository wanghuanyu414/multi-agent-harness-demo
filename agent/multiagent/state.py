"""Supervisor + Worker 多 Agent 共享状态定义"""
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _accumulate_or_reset(current: list | None, new: list | None) -> list:
    """fan-out 累积 + 每轮重置：空列表是「重置」信号，非空列表追加到当前。

    不能用 operator.add：`existing + [] == existing`，supervisor 每轮返回 [] 想清空
    旧结果会失效，导致历史结果跨轮泄漏；且 worker 回写整列表时旧值被反复叠加翻倍。
    约定：Supervisor 每轮返回 [] 清空；Worker 返回 [自己的 delta] 追加。
    """
    if not new:
        return []
    return list(current or []) + list(new)


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
    agents_to_dispatch: list  # 本轮要派发的 Worker 列表（并行 fan-out 用）
    worker_results: Annotated[list, _accumulate_or_reset]  # 本轮各 Worker 回复 [{"agent": ..., "response": ...}]（每轮重置）
    tool_results: Annotated[list, _accumulate_or_reset]  # 本轮各 Worker 实际调用的工具原始结果（供输出护栏事实性校验）
    plan_reasoning: str  # Supervisor 的派发推理（CoT）
    human_review: dict  # 人工审核结果 {"status": approved/rejected/skipped, "intent": ...}
    input_guardrail_result: dict  # 输入护栏结果 {"passed": bool, "violations": [...], "mode": ...}
    output_guardrail_result: dict  # 输出护栏结果 {"passed": bool, "violations": [...]}
