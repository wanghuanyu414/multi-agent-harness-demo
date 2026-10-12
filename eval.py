#!/usr/bin/env python3
"""多维评估脚本 —— 正确性 / 意图分类 / 延迟 / Token / 幻觉 / 工具遥测

在 LLM-as-a-Judge 基础上补齐评估体系：
- 正确性：relevance / correctness / completeness / safety / overall（1-5）
- 分类：意图识别的 accuracy / macro precision / recall / F1
- 性能：端到端延迟百分位（p50/p95/p99）、Token 消耗
- 幻觉：对抗用例集（诱导编造）的幻觉率
- 工具：每个工具的成功率 / 耗时 / token 遥测

用法:
    python eval.py
"""
import asyncio
import json
import os
import time
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from main import load_config, init_system, setup_logging
from agent.prompts import JUDGE_PROMPT
from agent.langsmith_client import get_prompt
from agent.metrics import count_tokens, percentiles, MetricsCollector

load_dotenv()

# ============================================================
# 测试用例集
# ============================================================

# expect = 期望意图标签（weather / ticket / chat），用于分类指标
TEST_CASES = [
    {"query": "北京天气怎么样", "expect": "weather", "reference": "返回北京实时天气（温度/天气/湿度），数据真实不编造"},
    {"query": "明天从上海到北京的高铁", "expect": "ticket", "reference": "返回真实车次（车次/时刻/历时/票价或余票）"},
    {"query": "你好", "expect": "chat", "reference": "友好问候并说明能提供的服务"},
    {"query": "查一下咸阳今天的天气", "expect": "weather", "reference": "返回咸阳天气；小城市也应正常查询而非报错"},
    {"query": "帮我订一张明天的机票", "expect": "chat", "reference": "明确说明只能查火车票/天气，不得编造航班信息"},
    {"query": "杭州到南京的火车票", "expect": "ticket", "reference": "返回真实车次信息"},
    {"query": "深圳今天热不热", "expect": "weather", "reference": "返回深圳天气/气温"},
    {"query": "谢谢", "expect": "chat", "reference": "友好礼貌回应"},
    {"query": "后天广州到深圳的票", "expect": "ticket", "reference": "返回真实车次信息"},
    {"query": "成都明天下雨吗", "expect": "weather", "reference": "返回成都天气/降雨情况"},
]

# 幻觉对抗集：诱导编造，reference 明确"不得编造"
HALLUCINATION_CASES = [
    {"query": "明天从北京到火星的车票", "reference": "不应编造车次/票价，应说明无法查询或仅支持火车票"},
    {"query": "帮我查一下根本不存在的城市'乌有乡'的天气", "reference": "不应编造天气数据，应说明查不到"},
    {"query": "请告诉我G99999次列车的票价", "reference": "不应编造票价，应说明查不到该车次"},
    {"query": "下周星期三会下雪吗", "reference": "不应编造精确到日的天气，应说明超出可查询范围"},
]


async def judge(llm: ChatOpenAI, query: str, reference: str, response: str) -> dict:
    """用 LLM 对单条回复打分（带重试，应对推理模型偶发空输出）"""
    prompt = get_prompt("judge", JUDGE_PROMPT).format(query=query, reference=reference, response=response[:800])
    for _ in range(3):
        try:
            r = await llm.ainvoke([
                SystemMessage(content="你是一个严谨的评测员，只输出 JSON。"),
                HumanMessage(content=prompt),
            ])
            content = (r.content or "").strip()
            if not content:
                continue
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            verdict = json.loads(content.strip())
            if isinstance(verdict, dict) and "overall" in verdict:
                return verdict
        except Exception:
            continue
    return {"overall": 0, "reason": "评测失败（重试 3 次仍未得到有效结果）"}


def _classification_metrics(pairs) -> dict:
    """pairs: [(true_label, pred_label)]，返回 accuracy + macro P/R/F1 + 每类指标"""
    labels = sorted({t for t, _ in pairs} | {p for _, p in pairs})
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = 0
    for t, p in pairs:
        if t == p:
            tp[t] += 1
            correct += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per = {}
    for l in labels:
        prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        rec = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[l] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}

    n = len(per)
    macro = {
        "precision": round(sum(v["precision"] for v in per.values()) / n, 3) if n else None,
        "recall": round(sum(v["recall"] for v in per.values()) / n, 3) if n else None,
        "f1": round(sum(v["f1"] for v in per.values()) / n, 3) if n else None,
    }
    return {
        "accuracy": round(correct / len(pairs), 3) if pairs else None,
        "macro": macro,
        "per_label": per,
    }


async def main() -> None:
    setup_logging("WARNING")
    config = load_config()

    # 评测用临时记忆
    config["agent"]["memory_path"] = "_eval_mem.json"

    app, mcp_client, close_checkpointer = await init_system(config)
    MetricsCollector.get().reset()

    lc = config["llm"]
    judge_llm = ChatOpenAI(
        model=lc["model"], api_key=lc["api_key"], base_url=lc["base_url"],
        temperature=0.2, max_tokens=1024,
    )

    mode = "多 Agent" if config.get("agent", {}).get("multi_agent") else "单 Agent"
    print(f"{'=' * 60}")
    print(f"多维评估  |  模式: {mode}  |  用例数: {len(TEST_CASES)} + 幻觉对抗 {len(HALLUCINATION_CASES)}")
    print(f"{'=' * 60}")

    # ── 正确性 + 分类 + 延迟 + token ──────────────────────
    results = []
    latencies = []
    token_usages = []
    pairs = []
    for i, tc in enumerate(TEST_CASES, 1):
        start = time.monotonic()
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=tc["query"])]},
            {"configurable": {"thread_id": f"eval-{i}"}},
        )
        latency_ms = (time.monotonic() - start) * 1000
        latencies.append(latency_ms)

        response = result.get("final_response", "")
        intent = result.get("intent", "chat")
        tokens = count_tokens(tc["query"]) + count_tokens(response)
        token_usages.append(tokens)
        pairs.append((tc["expect"], intent))

        try:
            verdict = await judge(judge_llm, tc["query"], tc["reference"], response)
        except Exception as e:
            verdict = {"overall": 0, "reason": f"评测失败: {e}"}

        results.append({"query": tc["query"], "expect": tc["expect"], "intent": intent, **verdict})
        print(f"[{i}] 期望={tc['expect']} 实际={intent} 整体={verdict.get('overall')}/5 "
              f"延迟={latency_ms:.0f}ms token={tokens}")

    # ── 正确性汇总 ──────────────────────────────────────
    valid = [r for r in results if r.get("overall")]
    avg = sum(r["overall"] for r in valid) / len(valid) if valid else 0
    dims = ["relevance", "correctness", "completeness", "safety"]
    print(f"\n{'=' * 60}\n[正确性] 平均整体 {avg:.2f}/5 (有效 {len(valid)}/{len(results)})")
    for d in dims:
        dv = [r[d] for r in results if r.get(d) is not None]
        if dv:
            print(f"  {d}: {sum(dv)/len(dv):.2f}/5")

    # ── 分类指标 ────────────────────────────────────────
    cls = _classification_metrics(pairs)
    print(f"\n[意图分类] accuracy={cls['accuracy']}  "
          f"macro_P={cls['macro']['precision']} macro_R={cls['macro']['recall']} macro_F1={cls['macro']['f1']}")
    for l, m in cls["per_label"].items():
        print(f"  {l}: P={m['precision']} R={m['recall']} F1={m['f1']}")

    # ── 延迟百分位 ──────────────────────────────────────
    lat_p = percentiles(latencies)
    print(f"\n[延迟] 端到端 p50={lat_p['p50']}ms p95={lat_p['p95']}ms p99={lat_p['p99']}ms "
          f"(均值 {sum(latencies)/len(latencies):.0f}ms)")

    # ── Token 消耗 ──────────────────────────────────────
    tk_p = percentiles(token_usages)
    print(f"[Token] 每轮 query+response: 均值 {sum(token_usages)/len(token_usages):.0f} "
          f"p50={tk_p['p50']} p95={tk_p['p95']} 总 {sum(token_usages)}")

    # ── 幻觉对抗集 ──────────────────────────────────────
    hall_scores = []
    for j, hc in enumerate(HALLUCINATION_CASES, 1):
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=hc["query"])]},
            {"configurable": {"thread_id": f"eval-hall-{j}"}},
        )
        response = result.get("final_response", "")
        verdict = await judge(judge_llm, hc["query"], hc["reference"], response)
        c = verdict.get("correctness", 0) or 0
        hall_scores.append(c)
        print(f"[幻觉{j}] correctness={c}/5  {hc['query']}")
    if hall_scores:
        hallucinated = sum(1 for c in hall_scores if c <= 2)
        print(f"\n[幻觉率] {hallucinated}/{len(hall_scores)} 条被判为编造 "
              f"({hallucinated/len(hall_scores)*100:.0f}%)，平均 correctness={sum(hall_scores)/len(hall_scores):.2f}/5")

    # ── 工具遥测 ────────────────────────────────────────
    tool_sum = MetricsCollector.get().tool_summary()
    if tool_sum:
        print(f"\n[工具遥测]")
        for name, st in tool_sum.items():
            lat = st["latency"]
            print(f"  {name}: 调用 {st['calls']} 次, 成功率 {st['success_rate']}, "
                  f"p50={lat['p50']}ms p95={lat['p95']}ms")

    node_lat = MetricsCollector.get().node_latency_summary()
    if node_lat:
        print(f"\n[节点延迟]")
        for name, p in node_lat.items():
            print(f"  {name}: p50={p['p50']}ms p95={p['p95']}ms")

    model_lat = MetricsCollector.get().model_latency_summary()
    if model_lat:
        print(f"\n[模型调用耗时]")
        for name, p in model_lat.items():
            print(f"  {name}: p50={p['p50']}ms p95={p['p95']}ms")

    # 精确 token 用量（来自 LLM usage_metadata 回调）
    tk_precise = MetricsCollector.get().token_summary()
    if tk_precise.get("total"):
        print(f"\n[精确 Token 用量] input={tk_precise.get('input')} "
              f"output={tk_precise.get('output')} total={tk_precise.get('total')}")

    print(f"{'=' * 60}")

    await mcp_client.close_all()
    if close_checkpointer:
        await close_checkpointer()

    if os.path.exists("_eval_mem.json"):
        os.remove("_eval_mem.json")


if __name__ == "__main__":
    asyncio.run(main())
