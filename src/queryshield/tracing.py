"""QueryShield's own spans, exported over OTLP/HTTP only when QUERYSHIELD_OTLP_TRACES_URL is set.

A span takes over the ids C20 already sends rather than making its own: every span of a run is on
the run's trace-id (``providers.http.run_trace_id``), and an HTTP attempt's span id is the
parent-id of its ``traceparent`` header, so the gateway's spans hang under it.  Spans carry
only the FIELDS below.  Off, the tracer is the API's no-op one: nothing is built, no thread
runs, and the global provider is never set, so the MCP SDK's own spans stay no-op either way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import os
import threading
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

URL_VARIABLE = "QUERYSHIELD_OTLP_TRACES_URL"
EXPORT_TIMEOUT_SECONDS = 5.0
SHUTDOWN_SECONDS = 5.0
# The only fields a span or a span event carries: ids, enums and numbers, never text a person
# or a model wrote (docs/operations.md).  Each becomes ``queryshield.<field>`` unless named here.
FIELDS = frozenset({
    "run_id", "segment", "profile", "approval_id", "model_call_id", "request_id", "agent", "subtask_index",
    "endpoint", "tool_name", "skill_id", "status", "error_code", "provider_error_code", "http_status",
    "attempt_count", "row_count", "model_call_count", "tool_call_count", "sql_exec_count",
    "prompt_tokens", "completion_tokens", "total_tokens",
})
_NAMES = {"http_status": "http.response.status_code"}

_tracer: trace.Tracer = trace.NoOpTracer()
_provider: TracerProvider | None = None
# The ids the span being started takes over: (trace-id, span-id), hex, either may be None.
_adopted: ContextVar[tuple[str | None, str | None]] = ContextVar("queryshield_adopted_ids", default=(None, None))


class _AdoptedIds(RandomIdGenerator):
    def generate_trace_id(self) -> int:
        trace_id = _adopted.get()[0]
        return int(trace_id, 16) if trace_id else super().generate_trace_id()

    def generate_span_id(self) -> int:
        span_id = _adopted.get()[1]
        return int(span_id, 16) if span_id else super().generate_span_id()


def configured_url() -> str:
    """The traces address the service is configured with; empty means off."""

    return os.getenv(URL_VARIABLE, "").strip()


def start(url: str) -> None:
    """Export to ``url`` (already checked) in batches; a collector that fails only loses spans."""

    _use(BatchSpanProcessor(OTLPSpanExporter(endpoint=url, timeout=EXPORT_TIMEOUT_SECONDS)))


def _use(processor: SpanProcessor) -> None:
    global _tracer, _provider
    # Sampler and resource are fixed here, not read from OTEL_* variables.  The exit flush is stop()'s, bounded.
    _provider = TracerProvider(
        sampler=ALWAYS_ON, resource=Resource({"service.name": "queryshield"}), id_generator=_AdoptedIds(), shutdown_on_exit=False
    )
    _provider.add_span_processor(processor)
    _tracer = _provider.get_tracer("queryshield")


def stop() -> None:
    """Send the remaining spans, waiting at most SHUTDOWN_SECONDS; the SDK's own shutdown takes no limit."""

    global _tracer, _provider
    provider, _provider, _tracer = _provider, None, trace.NoOpTracer()
    if provider is not None:
        closing = threading.Thread(target=provider.shutdown, name="queryshield-trace-shutdown", daemon=True)
        closing.start()
        closing.join(SHUTDOWN_SECONDS)


@contextmanager
def span(
    name: str,
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    """A span under the current one; with ``trace_id``, a root on that trace unless the current span is on it.

    An exception only marks the span failed: its message is never recorded (it can hold SQL or text).
    """

    current = trace.get_current_span().get_span_context()
    parent = None if trace_id is None or format(current.trace_id, "032x") == trace_id else otel_context.Context()
    token = _adopted.set((trace_id, span_id))
    try:
        started = _tracer.start_span(name, context=parent, kind=kind, attributes=attributes)
    finally:
        _adopted.reset(token)
    with trace.use_span(started, end_on_exit=True, record_exception=False, set_status_on_exception=False):
        try:
            yield started
        except BaseException:
            started.set_status(Status(StatusCode.ERROR))
            raise


def annotate(target: Span, fields: Mapping[str, object]) -> None:
    """Set the FIELDS among ``fields``; a failed status marks the span failed."""

    target.set_attributes(_attributes(fields))
    if fields.get("status") in {"failed", "FAILED"}:
        target.set_status(Status(StatusCode.ERROR))


def add_event(target: Span, name: str, fields: Mapping[str, object]) -> None:
    target.add_event(name, _attributes(fields))


def _attributes(fields: Mapping[str, object]) -> dict[str, Any]:
    return {_NAMES.get(key, f"queryshield.{key}"): value for key, value in fields.items() if key in FIELDS and value is not None}


@contextmanager
def detached() -> Iterator[None]:
    """No span is current inside: for work handed to a thread that must not take the run's trace along."""

    token = otel_context.attach(otel_context.Context())
    try:
        yield
    finally:
        otel_context.detach(token)


def carried(call: Callable[[], Any]) -> Callable[[], Any]:
    """``call`` in another thread under the current span.

    Only the OTel context goes along: a full contextvars copy would also hand LangGraph's own
    context to the sub-agent's graph.
    """

    captured = otel_context.get_current()

    def run() -> Any:
        token = otel_context.attach(captured)
        try:
            return call()
        finally:
            otel_context.detach(token)

    return run
