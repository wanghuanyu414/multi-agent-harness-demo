#!/usr/bin/env python3
"""旅行助手 Agent 启动脚本 —— 环境检查与依赖安装"""
import subprocess
import sys
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent


def check_python() -> bool:
    """检查 Python 版本"""
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        print(f"  ✅ Python {major}.{minor}")
        return True
    print(f"  ❌ Python {major}.{minor} (需要 3.10+)")
    return False


def check_npx() -> bool:
    """检查 npx 是否可用"""
    try:
        result = subprocess.run(
            ["npx", "--version"], capture_output=True, text=True, timeout=10,
            shell=True,
        )
        if result.returncode == 0:
            print(f"  ✅ npx {result.stdout.strip()}")
            return True
    except Exception:
        pass
    print("  ⚠️  npx 未检测到（MCP 服务器需要 Node.js/npm）")
    return False  # 不阻塞，因为可能是可选依赖


def check_env() -> bool:
    """检查 .env 文件"""
    env_file = PROJECT_DIR / ".env"
    env_example = PROJECT_DIR / ".env.example"

    if env_file.exists():
        print("  ✅ .env 文件已存在")
        return True

    if env_example.exists():
        print("  ⚠️  未找到 .env 文件，正在从 .env.example 创建...")
        env_file.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
        print("  📝 请编辑 .env 文件填入你的 OPENAI_API_KEY")
        return False

    print("  ❌ 未找到 .env.example 文件")
    return False


def install_deps() -> bool:
    """安装 Python 依赖"""
    req_file = PROJECT_DIR / "requirements.txt"
    if not req_file.exists():
        print("  ❌ 未找到 requirements.txt")
        return False

    print("  📦 正在安装依赖...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            check=True,
            cwd=str(PROJECT_DIR),
        )
        print("  ✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError:
        print("  ❌ 依赖安装失败，请手动运行: pip install -r requirements.txt")
        return False


def main() -> None:
    print("=" * 50)
    print("  🚄 旅行助手 Agent 环境检查")
    print("=" * 50)
    print()

    print("[1/4] Python 版本:")
    py_ok = check_python()

    print("\n[2/4] NPX (Node.js):")
    npx_ok = check_npx()

    print("\n[3/4] 环境配置:")
    env_ok = check_env()

    print("\n[4/4] Python 依赖:")
    dep_ok = install_deps()

    print("\n" + "=" * 50)
    if py_ok and dep_ok:
        print("  ✅ 环境就绪，可以启动程序")
        print("  💡 运行: python main.py")
        if not env_ok:
            print("  ⚠️  请先编辑 .env 文件配置 API Key")
        if not npx_ok:
            print("  ⚠️  MCP 服务器功能需要安装 Node.js")
    else:
        print("  ❌ 环境检查未通过，请解决上述问题后重试")
    print("=" * 50)


if __name__ == "__main__":
    main()