"""Checkpointer 工厂 —— 支持 redis / sqlite / postgres / memory 四种后端。"""
import logging

logger = logging.getLogger(__name__)


async def build_checkpointer(config: dict):
    """按配置构建 checkpointer，返回 (checkpointer, close)。

    close 为可选异步清理协程（持久化连接用）；内存后端为 None。
    """
    cp = (config.get("agent") or {}).get("checkpoint") or {}
    backend = str(cp.get("backend", "sqlite")).lower()

    if backend == "redis":
        try:
            from langgraph.checkpoint.redis import AsyncRedisSaver
        except ImportError:
            logger.warning("langgraph-checkpoint-redis 未安装，回退内存 checkpointer")
        else:
            redis_url = cp.get("redis_url", "redis://localhost:6379")
            cm = AsyncRedisSaver.from_conn_string(redis_url)
            saver = await cm.__aenter__()
            await saver.setup()
            logger.info("checkpointer: redis (%s)", redis_url)
            return saver, _make_close(cm)

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError:
            logger.warning("langgraph-checkpoint-postgres 未安装，回退内存 checkpointer")
        else:
            conn_string = cp.get("conn_string", "")
            if conn_string:
                cm = AsyncPostgresSaver.from_conn_string(conn_string)
                saver = await cm.__aenter__()
                await saver.setup()
                logger.info("checkpointer: postgres")
                return saver, _make_close(cm)
            logger.warning("postgres checkpointer 缺少 conn_string，回退内存 checkpointer")

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError:
            logger.warning("langgraph-checkpoint-sqlite 未安装，回退内存 checkpointer")
        else:
            path = cp.get("path", "checkpoints.db")
            cm = AsyncSqliteSaver.from_conn_string(path)
            saver = await cm.__aenter__()
            await saver.setup()
            logger.info("checkpointer: sqlite (%s)", path)
            return saver, _make_close(cm)

    from langgraph.checkpoint.memory import InMemorySaver
    logger.info("checkpointer: memory（非持久化，进程重启丢失）")
    return InMemorySaver(), None


def _make_close(cm):
    async def close():
        try:
            await cm.__aexit__(None, None, None)
        except Exception as e:
            logger.debug("checkpointer 关闭失败: %s", e)
    return close
