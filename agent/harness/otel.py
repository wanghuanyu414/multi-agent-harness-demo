"""OpenTelemetry 集成 —— 标准 trace 出口。

把自研 span 追踪升级为 OTel 标准：trace_id/span_id 自动贯穿，
可上报到 OTLP Collector（Jaeger/Tempo），或 console 调试。

配置（config.yaml → otel）：
  enabled: false        # 关闭时用 no-op tracer（零开销）
  service_name: travel-agent
  endpoint: ""          # OTLP HTTP endpoint；留空用 console exporter
"""
import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

logger = logging.getLogger("otel")

_tracer = None


def init_otel(config: dict | None = None) -> None:
    """初始化 OTel tracer provider（进程启动时调用一次）。"""
    global _tracer
    cfg = config or {}
    if not cfg.get("enabled", False):
        _tracer = trace.get_tracer(__name__)
        logger.info("OTel 未启用（no-op tracer）")
        return

    service_name = cfg.get("service_name", "travel-agent")
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))

    endpoint = cfg.get("endpoint", "")
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        logger.info("OTel 启用: service=%s exporter=otlp(%s)", service_name, endpoint)
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OTel 启用: service=%s exporter=console", service_name)

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


def get_tracer():
    """返回 tracer（未 init_otel 时懒初始化为 no-op）。"""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(__name__)
    return _tracer
