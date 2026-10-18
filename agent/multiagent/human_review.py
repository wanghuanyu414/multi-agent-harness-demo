"""人工审核节点 —— 最终回复返回用户前，对指定意图做 Human-in-the-loop 审核"""
import logging

from langgraph.graph import END
from langgraph.types import Command, interrupt

logger = logging.getLogger(__name__)

_REJECT_FALLBACK = "抱歉，该回复未通过人工审核，已被取消。"


def make_human_review_node(config: dict, agent_to_intent: dict | None = None):
    """构造人工审核节点。

    用 LangGraph 的 interrupt() 在最终回复返回用户前挂起，等待人工 approve / edit / reject。
    审核范围由 config 决定：
    - enabled: 是否开启
    - intents: 审核哪些意图，["all"] 表示全部；例如 ["ticket"] 只审车票

    决策值（由 Command(resume=...) 传入）约定为：
    - {"approve": True}                通过，返回原回复
    - {"approve": True, "response": x} 通过并采用人工修改后的 x
    - {"approve": False}               拒绝，返回兜底话术
    """
    enabled = bool(config.get("enabled", False))
    intents = config.get("intents", [])

    async def human_review_node(state) -> Command:
        intent = state.get("intent", "chat")
        state_intents = state.get("intents") or [intent]  # 单 Agent 多意图集合
        response = state.get("final_response", "")

        if agent_to_intent:
            dispatched_intents = {
                agent_to_intent.get(a) for a in state.get("agents_to_dispatch", [])
            }
            dispatched_intents.discard(None)
            needs_review = enabled and (
                intents == ["all"] or bool(dispatched_intents & set(intents))
            )
        else:
            needs_review = enabled and (
                intents == ["all"] or any(i in intents for i in state_intents)
            )

        if not needs_review:
            return Command(update={"human_review": {"status": "skipped", "intent": intent}}, goto=END)

        # 挂起，等待人工审核；decision 是 Command(resume=...) 传入的值
        decision = interrupt({
            "type": "human_review",
            "intent": intent,
            "response": response,
        })

        if not isinstance(decision, dict):
            decision = {"approve": bool(decision)}

        if decision.get("approve", False):
            final = decision.get("response", response) or response
            status = "approved"
        else:
            final = _REJECT_FALLBACK
            status = "rejected"

        logger.info(f"人工审核完成: intent={intent}, status={status}")
        return Command(update={
            "final_response": final,
            "human_review": {"status": status, "intent": intent},
        }, goto=END)

    return human_review_node
