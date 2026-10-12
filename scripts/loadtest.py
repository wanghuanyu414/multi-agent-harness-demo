#!/usr/bin/env python3
"""最小压测脚本 —— 定位单进程 Web 层吞吐上限与瓶颈。

不依赖额外库（只用 httpx，langchain 已带）。

用法：
  # 1) 纯 Web 层（不花钱、不经过 LLM/MCP）：测事件循环 / 框架上限
  python scripts/loadtest.py -c 100 -n 5000

  # 2) 端到端（真实 LLM，会花钱且受上游 RPM/TPM 限流）：测端到端延迟 + 429 率
  python scripts/loadtest.py --chat -c 20 -n 100

  # 3) 天气意图（测并发下天气工具是否还阻塞）
  python scripts/loadtest.py --chat "北京天气怎么样" -c 20 -n 100

先启动服务：python web.py（默认 127.0.0.1:8092）
"""
import argparse
import asyncio
import time

import httpx


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


async def worker(client, sem, url, method, json_body, results):
    async with sem:
        start = time.perf_counter()
        try:
            if method == "POST":
                resp = await client.post(url, json=json_body)
            else:
                resp = await client.get(url)
            status = resp.status_code
            await resp.aread()
        except Exception:
            status = -1  # 连接失败/超时
        results.append((status, (time.perf_counter() - start) * 1000))


async def run(url, method, json_body, concurrency, total):
    results = []
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_connections=concurrency + 10,
        max_keepalive_connections=concurrency,
    )
    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:
        tasks = [asyncio.create_task(worker(client, sem, url, method, json_body, results))
                 for _ in range(total)]
        await asyncio.gather(*tasks)
    return results


def summarize(results, wall_seconds):
    statuses = {}
    ok_lat = []      # 200 的延迟
    for status, ms in results:
        statuses[status] = statuses.get(status, 0) + 1
        if status == 200:
            ok_lat.append(ms)
    ok_lat.sort()

    print(f"\n{'=' * 56}")
    print(f"总请求: {len(results)}   总耗时: {wall_seconds:.2f}s   QPS: {len(results) / wall_seconds:.1f}")
    print(f"状态码分布: {statuses}")
    if ok_lat:
        avg = sum(ok_lat) / len(ok_lat)
        print(f"延迟(仅200, ms): min={ok_lat[0]:.0f}  avg={avg:.0f}  "
              f"P50={percentile(ok_lat, 0.50):.0f}  P95={percentile(ok_lat, 0.95):.0f}  "
              f"P99={percentile(ok_lat, 0.99):.0f}  max={ok_lat[-1]:.0f}")
    n429 = statuses.get(429, 0)
    if n429:
        print(f"⚠️  429 限流: {n429}/{len(results)}（{n429 / len(results) * 100:.0f}%）—— 说明撞到本机/上游限流")
    print(f"{'=' * 56}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message", nargs="?", default=None, help="chat 消息内容（配合 --chat）")
    ap.add_argument("--base", default="http://127.0.0.1:8092", help="服务地址")
    ap.add_argument("--chat", action="store_true", help="打 /api/chat（真实 LLM）；默认打 /health")
    ap.add_argument("--url", default="/health", help="自定义路径（未用 --chat 时生效）")
    ap.add_argument("-c", "--concurrency", type=int, default=50, help="并发数")
    ap.add_argument("-n", "--requests", type=int, default=1000, help="总请求数")
    args = ap.parse_args()

    if args.chat:
        url = args.base + "/api/chat"
        method, body = "POST", {"message": args.message or "你好"}
    else:
        url = args.base + args.url
        method, body = "GET", None

    print(f"目标: {method} {url}  并发={args.concurrency}  总数={args.requests}")
    start = time.perf_counter()
    results = asyncio.run(run(url, method, body, args.concurrency, args.requests))
    summarize(results, time.perf_counter() - start)


if __name__ == "__main__":
    main()
