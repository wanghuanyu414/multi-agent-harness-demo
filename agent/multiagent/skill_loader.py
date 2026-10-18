"""SKILL.md 加载器 —— 扫描 skills/ 目录，解析 SKILL.md 为 Skill 配置 dict。

只做文件解析，不 import skills.py（避免循环依赖）；返回 raw dict，
由 skills.py 构造 Skill 对象。
"""
import asyncio
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _parse_skill_md(text: str) -> tuple[dict, str]:
    """解析 SKILL.md：返回 (frontmatter dict, 正文 str)。

    frontmatter 用两个 `---` 包裹（YAML），其后是正文（system prompt）。
    """
    text = text.strip()
    if not text.startswith("---"):
        raise ValueError("SKILL.md 缺少 YAML frontmatter（应以 --- 开头）")
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise ValueError("SKILL.md frontmatter 未闭合（缺少结尾的 ---）")

    frontmatter_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1:]).strip()
    frontmatter = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError("SKILL.md frontmatter 必须是 YAML 映射")
    return frontmatter, body


def load_skill_files(skills_dir: Path) -> list[dict]:
    """扫描 skills_dir 下每个 */SKILL.md，解析 frontmatter（第 1 级元数据）。

    正文（system prompt，第 2 级）不在此加载，由 Skill.prompt 惰性读取。
    返回 [{frontmatter 字段..., "_skill_md_path": Path}]。
    """
    skills_dir = Path(skills_dir)
    result: list[dict] = []
    if not skills_dir.is_dir():
        logger.warning(f"skills 目录不存在: {skills_dir}")
        return result

    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            text = skill_md.read_text(encoding="utf-8")
            frontmatter, body = _parse_skill_md(text)
            if not frontmatter.get("name"):
                raise ValueError("frontmatter 缺少 name 字段")
            if not body:
                raise ValueError("正文（system prompt）为空")
            raw = dict(frontmatter)
            raw["_skill_md_path"] = skill_md
            result.append(raw)
            logger.info(f"加载 skill 元数据: {frontmatter.get('name')} ({skill_md.parent.name})")
        except Exception as e:
            raise ValueError(f"解析 {skill_md} 失败: {e}") from e
    return result


def load_skill_prompt(skill_md_path: Path) -> str:
    """按需加载 SKILL.md 正文（第 2 级，system prompt）。"""
    text = Path(skill_md_path).read_text(encoding="utf-8")
    _, body = _parse_skill_md(text)
    return body


async def run_skill_script(skill_md_path: Path, script_name: str, input_text: str) -> str | None:
    """执行 skill 目录 scripts/ 下的脚本（标准 SKILL.md 第 3 级按需执行）。

    通过沙箱执行（backend 由 config 控制：docker / subprocess / none），
    脚本代码不加载进主进程，只有 stdout 进入上下文。
    返回 stdout 文本；脚本缺失或执行失败返回 None（调用方回退原始结果）。
    """
    script = Path(skill_md_path).parent / "scripts" / script_name
    if not script.is_file():
        return None
    from ..harness.sandbox import SandboxError, get_sandbox

    sandbox = get_sandbox()
    try:
        if sandbox is not None:
            stdout, _ = await sandbox.run(script, input_text)
        else:
            stdout = await _run_direct(script, input_text)
        return stdout
    except SandboxError as e:
        logger.warning(f"skill 脚本 {script_name} 执行失败: {e}")
    except Exception as e:
        logger.warning(f"skill 脚本 {script_name} 执行异常: {e}")
    return None


async def _run_direct(script: Path, input_text: str) -> str:
    """无沙箱时直接子进程执行（仅开发用，backend=none）"""
    from ..harness.sandbox import SandboxError

    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input_text.encode("utf-8"))
    if proc.returncode != 0:
        raise SandboxError(
            f"脚本退出码 {proc.returncode}: {stderr.decode('utf-8', errors='replace')[:200]}"
        )
    return stdout.decode("utf-8", errors="replace")
