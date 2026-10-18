"""SKILL 脚本执行沙箱 —— 可插拔后端，配置开关控制。

后端（config.yaml → sandbox.backend）：
- docker     ：Docker 容器沙箱（真隔离：文件系统/网络/资源），不可用时降级 subprocess
- subprocess ：受限子进程沙箱（弱隔离，但无 Docker 依赖、快）
- none       ：无沙箱，直接子进程裸跑（仅开发用）

协议 SkillSandbox.run(script_host_path, input_text) -> (stdout, stderr)。
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("harness.sandbox")


class SandboxError(Exception):
    """沙箱执行失败（超时 / 非零退出 / 容器启动失败）"""


class SkillSandbox(Protocol):
    """沙箱协议：所有后端实现 run 方法"""

    async def run(self, script_host_path: Path, input_text: str) -> tuple[str, str]:
        ...


class SubprocessSandbox:
    """受限子进程沙箱：弱隔离（只有超时），但无 Docker 依赖、快"""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def run(self, script_host_path: Path, input_text: str) -> tuple[str, str]:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_host_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input_text.encode("utf-8")), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise SandboxError(f"脚本执行超时（{self.timeout}s）")
        if proc.returncode != 0:
            raise SandboxError(
                f"脚本退出码 {proc.returncode}: {stderr.decode('utf-8', errors='replace')[:200]}"
            )
        return stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


class DockerSandbox:
    """Docker 容器沙箱：真隔离（文件系统 / 网络 / 资源 / 非 root），不可用时降级子进程"""

    def __init__(self, image: str = "python:3.11-alpine", memory: str = "64m",
                 cpus: str = "0.5", pids_limit: int = 64, timeout: float = 30.0):
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.timeout = timeout
        self._fallback = SubprocessSandbox(timeout)

    async def run(self, script_host_path: Path, input_text: str) -> tuple[str, str]:
        script_host_path = Path(script_host_path)
        try:
            return await self._run_docker(script_host_path, input_text)
        except (FileNotFoundError, SandboxError) as e:
            logger.warning(f"Docker 沙箱不可用，降级到子进程（弱隔离）: {e}")
            return await self._fallback.run(script_host_path, input_text)

    async def _run_docker(self, script_host_path: Path, input_text: str) -> tuple[str, str]:
        host_path = script_host_path.resolve().as_posix()
        cmd = [
            "docker", "run", "--rm", "-i",
            "--network=none",
            f"--memory={self.memory}",
            f"--cpus={self.cpus}",
            f"--pids-limit={self.pids_limit}",
            "--read-only", "--tmpfs", "/tmp",
            "--user", "65534:65534",
            "-v", f"{host_path}:/script.py:ro",
            self.image,
            "python", "-B", "/script.py",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input_text.encode("utf-8")), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise SandboxError(f"脚本执行超时（{self.timeout}s）")
        if proc.returncode != 0:
            raise SandboxError(
                f"脚本退出码 {proc.returncode}: {stderr.decode('utf-8', errors='replace')[:200]}"
            )
        return stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


# 单例
_sandbox: SkillSandbox | None = None


def init_sandbox(config: dict | None = None) -> SkillSandbox | None:
    """从 config 初始化沙箱单例（进程启动时调用一次）。

    sandbox.backend 决定后端：docker / subprocess / none。
    """
    global _sandbox
    cfg = (config or {}).get("sandbox", {}) or {}
    backend = str(cfg.get("backend", "docker"))
    timeout = float(cfg.get("timeout", 30))

    if backend in ("none", "off", "disabled") or not cfg.get("enabled", True):
        _sandbox = None
        logger.info("沙箱已禁用（脚本直接子进程裸跑，仅开发用）")
        return None

    if backend == "subprocess":
        _sandbox = SubprocessSandbox(timeout=timeout)
        logger.info("沙箱后端: subprocess（受限子进程，弱隔离）")
    elif backend == "docker":
        _sandbox = DockerSandbox(
            image=cfg.get("image", "python:3.11-alpine"),
            memory=cfg.get("memory", "64m"),
            cpus=str(cfg.get("cpus", 0.5)),
            pids_limit=int(cfg.get("pids_limit", 64)),
            timeout=timeout,
        )
        logger.info(f"沙箱后端: docker (image={_sandbox.image})")
    else:
        raise ValueError(f"未知 sandbox.backend: {backend}")

    return _sandbox


def get_sandbox() -> SkillSandbox | None:
    """返回沙箱单例；未初始化或禁用返回 None"""
    return _sandbox
