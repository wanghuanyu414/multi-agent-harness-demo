"""长期记忆 —— JSON 文件持久化，跨会话保留用户信息与偏好"""
import json
import logging
import os
import tempfile
import threading
import time

from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import FACT_EXTRACT_PROMPT
from .langsmith_client import get_prompt
from .llm_utils import ainvoke_with_retry, parse_llm_json

logger = logging.getLogger(__name__)


class LongTermMemory:
    """长期记忆存储

    以 JSON 文件持久化记忆条目，进程重启后仍保留。
    每条记忆：{"content": str, "kind": str, "metadata": dict, "timestamp": float}
    """

    def __init__(self, path: str = "memory_store.json"):
        self.path = path
        self._lock = threading.Lock()
        self.entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"记忆文件加载失败: {e}")
            return []

    def _save(self) -> None:
        """原子写：写临时文件后 os.replace。"""
        tmp_path = None
        try:
            dir_name = os.path.dirname(os.path.abspath(self.path)) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".mem_", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.path)
            tmp_path = None
        except Exception as e:
            logger.error(f"记忆文件保存失败: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def remember(self, content: str, kind: str = "fact", metadata: dict | None = None) -> None:
        """写入一条记忆（去重：相同 content 不重复写入）"""
        content = (content or "").strip()
        if not content:
            return
        with self._lock:
            for e in self.entries:
                if e.get("content") == content:
                    return
            self.entries.append({
                "content": content,
                "kind": kind,
                "metadata": metadata or {},
                "timestamp": time.time(),
            })
            self._save()
        logger.info(f"记忆已写入: {content}")

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """按关键词重叠 + 时间衰减召回最相关的记忆"""
        with self._lock:
            entries = list(self.entries)
        if not query or not entries:
            return []
        scored: list[tuple[float, dict]] = []
        now = time.time()
        for e in entries:
            score = self._score(query, e.get("content", ""))
            if score > 0:
                # 时间衰减：越新的记忆权重略高
                age_days = (now - e.get("timestamp", now)) / 86400.0
                score = score * (1.0 / (1.0 + age_days))
                scored.append((score, e))
        scored.sort(key=lambda x: (-x[0], -x[1].get("timestamp", 0)))
        return [e for _, e in scored[:top_k]]

    @staticmethod
    def _score(query: str, content: str) -> float:
        """基于字符 2-gram 重叠的打分（中文友好）"""
        def bigrams(s: str) -> set:
            return {s[i:i + 2] for i in range(len(s) - 1)}
        q = bigrams(query)
        c = bigrams(content)
        overlap = q & c
        if not overlap:
            return 0.0
        return len(overlap) / max(len(q), 1)

    def to_context(self, memories: list[dict]) -> str:
        """把记忆格式化为可注入 prompt 的文本"""
        if not memories:
            return ""
        return "\n".join(f"- {m['content']}" for m in memories)

    def clear(self) -> None:
        """清空记忆"""
        with self._lock:
            self.entries = []
            self._save()


# ============================================================
# 记忆固化：从对话中提取值得长期记住的用户事实
# ============================================================

async def extract_facts(llm, query: str, response: str) -> list[str]:
    """用 LLM 从一轮对话中提取可长期记住的事实，失败时返回空列表"""
    try:
        prompt = get_prompt("fact-extract", FACT_EXTRACT_PROMPT).format(query=query[:200], response=response[:300])
        r = await ainvoke_with_retry(llm, [
            SystemMessage(content="你是记忆固化模块，负责从对话中提取值得长期记住的用户信息。"),
            HumanMessage(content=prompt),
        ])
        data = parse_llm_json(r.content)
        if isinstance(data, list):
            return [str(x) for x in data if x and str(x).strip()]
    except Exception as e:
        logger.warning(f"记忆固化提取失败: {e}")
    return []
