"""The one renderer for server-verified answers (B1 graph, B0 single pass, approval).

The first line is the verified facts; with a phrase-table reading, one basis
note per metric follows (catalog name, the catalog phrase that grounds it, the
fixed premise).  Every string comes from the fact records or the catalog: no
model text and no user text is copied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from queryshield.catalog.phrases import ClarificationReading, metric_basis_note


def _field(fact: object, name: str) -> object:
    return fact.get(name) if isinstance(fact, Mapping) else getattr(fact, name, None)


def render_verified_answer(
    facts: Sequence[object],
    *,
    clarifications: ClarificationReading | None = None,
) -> str:
    rendered: list[str] = []
    metric_ids: list[str] = []
    for fact in facts:
        label = _field(fact, "label")
        display_value = _field(fact, "display_value")
        window = _field(fact, "time_window")
        metric_id = _field(fact, "metric_id")
        if (
            type(label) is not str
            or type(display_value) is not str
            or not isinstance(window, Mapping)
            or any(type(window.get(name)) is not str for name in ("start", "end", "timezone"))
        ):
            raise ValueError("resolved fact record is missing its display contract")
        rendered.append(f"{label}：{display_value}（{window['start']}至{window['end']}，{window['timezone']}）")
        if type(metric_id) is str and metric_id not in metric_ids:
            metric_ids.append(metric_id)
    answer = "已核实：" + "；".join(rendered)
    if clarifications is None or not metric_ids:
        return answer
    return answer + "\n" + "".join(metric_basis_note(clarifications, metric_id) for metric_id in metric_ids)


def render_no_data_answer(metric_names: Sequence[str]) -> str:
    """The fixed reply for a question that needs no data (B3c-2 basis no_data).

    Server text plus the catalog names of the declarable metrics, passed in by
    the caller; no model text and no user text.
    """

    names = [name for name in metric_names if type(name) is str and name]
    if not names:
        raise ValueError("no_data reply needs the catalog metric names")
    return "可以查询你权限范围内本租户的经营数据指标：" + "、".join(names) + "。请说明要查的指标和时间范围。"


__all__ = ["render_no_data_answer", "render_verified_answer"]
