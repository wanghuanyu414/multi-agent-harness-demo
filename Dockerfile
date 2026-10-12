FROM python:3.12-slim

WORKDIR /app

# 先复制依赖清单，利用 Docker 层缓存；mcp-server-12306 为 Windows 专用，跳过安装
COPY requirements.txt .
RUN grep -v '^mcp-server-12306' requirements.txt > requirements.linux.txt \
    && pip install --no-cache-dir -r requirements.linux.txt

# 复制项目（.dockerignore 已排除 .env / .venv / __pycache__ 等）
COPY . .

# 容器内默认用内存检查点（无 Redis 时）；可用 -e 覆盖
ENV CHECKPOINT_BACKEND=memory \
    PORT=8092

EXPOSE 8092

CMD ["python", "web.py"]
