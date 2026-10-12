#!/usr/bin/env python3
"""一次性脚本：把 agent/prompts.py 里的本地 prompt 常量上传到 LangSmith Hub。

用法:
    python scripts/push_prompts.py

幂等：prompt 已存在则更新。走 LangSmith SaaS，需 .env 中 LANGSMITH_ENABLED=true 且代理可用。
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 让脚本能 import 项目根目录的 main 与 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate

from main import load_config
from agent.langsmith_client import init_langsmith, get_client, PROMPT_NAMESPACE
from agent.prompts import PROMPTS


def main() -> None:
    config = load_config()
    init_langsmith(config.get("langsmith", {}))
    client = get_client()
    if client is None:
        print("LangSmith 未启用或初始化失败。请确认 LANGSMITH_ENABLED=true 且代理可用")
        sys.exit(1)

    ok = fail = 0
    for name, template in PROMPTS.items():
        identifier = f"{PROMPT_NAMESPACE}/{name}"
        prompt_obj = ChatPromptTemplate.from_messages([("system", template)])
        try:
            client.push_prompt(identifier, object=prompt_obj)
            print(f"  [ok] {identifier}")
            ok += 1
        except Exception as e:
            print(f"  [fail] {identifier}: {e}")
            fail += 1

    print(f"\n完成：成功 {ok}，失败 {fail}")


if __name__ == "__main__":
    main()
