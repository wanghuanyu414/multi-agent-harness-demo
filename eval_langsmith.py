#!/usr/bin/env python3
"""LangSmith 评测 —— 用 dataset + evaluate 跑回归评测

用法:
    python eval_langsmith.py --init   # 仅创建/更新 dataset
    python eval_langsmith.py          # 跑评测并上传到 LangSmith

复用 eval.py 的 TEST_CASES 与 LLM-as-a-Judge 逻辑；走 LangSmith SaaS，
需 .env 中 LANGSMITH_ENABLED=true 且代理可用。
"""
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import aevaluate

from main import load_config, init_system, setup_logging
from agent.langsmith_client import init_langsmith, get_client
from eval import TEST_CASES, judge

DATASET_NAME = "travel-agent-eval"


def _create_dataset(client) -> None:
    """创建/更新评测 dataset（幂等：已存在则复用同名 dataset）"""
    ds = client.create_dataset(DATASET_NAME, description="旅行助手回归评测用例")
    client.create_examples(
        inputs=[{"query": tc["query"]} for tc in TEST_CASES],
        outputs=[{"expect": tc["expect"], "reference": tc["reference"]} for tc in TEST_CASES],
        dataset_id=ds.id,
    )
    print(f"dataset '{DATASET_NAME}' 已就绪（{len(TEST_CASES)} 条）")


async def main() -> None:
    setup_logging("WARNING")
    config = load_config()
    init_langsmith(config.get("langsmith", {}))
    client = get_client()
    if client is None:
        print("LangSmith 未启用或初始化失败。请确认 LANGSMITH_ENABLED=true 且代理可用")
        sys.exit(1)

    if "--init" in sys.argv:
        _create_dataset(client)
        return

    # 评测用临时记忆
    config["agent"]["memory_path"] = "_eval_mem.json"
    app, mcp_client, close_checkpointer = await init_system(config)

    lc = config["llm"]
    judge_llm = ChatOpenAI(
        model=lc["model"],
        api_key=lc["api_key"],
        base_url=lc["base_url"],
        temperature=0.2,
        max_tokens=1024,
    )

    verdicts: list[dict] = []

    async def target(inputs: dict) -> dict:
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=inputs["query"])]},
            {"configurable": {"thread_id": "langsmith-eval"}},
        )
        return {"response": result.get("final_response", "")}

    async def judge_evaluator(run, example) -> dict:
        query = (example.inputs or {}).get("query", "")
        reference = (example.outputs or {}).get("reference", "")
        response = (run.outputs or {}).get("response", "")
        verdict = await judge(judge_llm, query, reference, response)
        overall = verdict.get("overall", 0) or 0
        verdicts.append(verdict)
        print(f"[{len(verdicts)}] overall={overall}/5  reason={verdict.get('reason', '')}")
        return {
            "key": "llm_judge",
            "score": overall / 5.0,
            "comment": verdict.get("reason", ""),
        }

    await aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[judge_evaluator],
        client=client,
        experiment_prefix="travel-agent",
        max_concurrency=1,
    )

    valid = [v for v in verdicts if v.get("overall")]
    if valid:
        avg = sum(v["overall"] for v in valid) / len(valid)
        print(f"\n汇总: 平均整体分 {avg:.2f}/5  (有效 {len(valid)}/{len(verdicts)})")

    await mcp_client.close_all()
    if close_checkpointer:
        await close_checkpointer()

    if os.path.exists("_eval_mem.json"):
        os.remove("_eval_mem.json")


if __name__ == "__main__":
    asyncio.run(main())
