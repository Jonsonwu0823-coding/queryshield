"""The catalog phrase table (catalog-v3 and later) as the server reads it.

This is the only place that matches business wording against the catalog.
The server never uses it to choose a metric: it checks a metric the model
declared (``check_declaration``), checks a clarification the model asked
(``review_ask``), resolves a user's answer to a clarification
(``select_value``) and names the basis of a verified answer
(``metric_basis_note``).

Matching: text is casefolded; explicit phrases are cut leftmost-longest
without overlap; an ambiguous phrase is covered when an explicit phrase's
span contains it (the "总额" inside "已支付订单总额").
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from queryshield.catalog.catalog import ClarificationRule, ClarificationValue, SemanticCatalog


@dataclass(frozen=True)
class PhraseHit:
    """One explicit phrase found in a text; ``phrase`` is the catalog spelling."""

    start: int
    end: int
    phrase: str
    metrics: frozenset[str]
    values: frozenset[tuple[str, str]]


def _scan(text: str, table: Mapping[str, object]) -> list[tuple[int, int, str]]:
    """Leftmost-longest, non-overlapping matches of ``table`` keys in casefolded ``text``."""

    keys = sorted(table, key=len, reverse=True)
    hits: list[tuple[int, int, str]] = []
    index = 0
    while index < len(text):
        match = next((key for key in keys if text.startswith(key, index)), None)
        if match is None:
            index += 1
            continue
        hits.append((index, index + len(match), match))
        index += len(match)
    return hits


class PhraseIndex:
    """The phrase table of one catalog."""

    def __init__(self, catalog: SemanticCatalog) -> None:
        self.catalog = catalog
        self.rules: tuple[ClarificationRule, ...] = tuple(rule for rule in catalog.clarifications if rule.values)
        # casefolded phrase -> (spelling, metrics named, (rule, value) pairs named)
        self._explicit: dict[str, tuple[str, frozenset[str], frozenset[tuple[str, str]]]] = {}
        for entry in catalog.entries:
            if entry.kind != "metric":
                continue
            metric_id = entry.id.removeprefix("metric.")
            values = frozenset(
                (rule.id, value.value) for rule in self.rules for value in rule.values if value.metric == metric_id
            )
            for phrase in catalog.metric_phrases(metric_id):
                self._explicit[phrase.casefold()] = (phrase, frozenset({metric_id}), values)
        for rule in self.rules:
            for value in rule.values:
                for phrase in value.phrases:
                    self._explicit[phrase.casefold()] = (phrase, frozenset(), frozenset({(rule.id, value.value)}))
        self._ambiguous = tuple((phrase.casefold(), rule.id) for rule in self.rules for phrase in rule.ambiguous_phrases)
        # Recognizing which rule a model's question is about (never the user's
        # text).  A metric phrase points at a rule only when that metric is one
        # of the rule's options ("退款后净额" asks about metric_basis, not about
        # the single-value refund window that net_fen implies).
        self._ask_terms: dict[str, frozenset[str]] = {}
        option_rules = {
            value.value: {rule.id for rule in self.rules if len(rule.values) > 1 and value in rule.values}
            for rule in self.rules
            for value in rule.values
            if value.metric == value.value
        }
        for key, (_, metrics, values) in self._explicit.items():
            if metrics:
                self._add_ask_term(key, {rule_id for metric in metrics for rule_id in option_rules.get(metric, ())})
            else:
                self._add_ask_term(key, {rule_id for rule_id, _ in values})
        for key, rule_id in self._ambiguous:
            self._add_ask_term(key, {rule_id})
        for rule in self.rules:
            for marker in rule.ask_markers:
                self._add_ask_term(marker.casefold(), {rule.id})

    def _add_ask_term(self, key: str, rule_ids: set[str]) -> None:
        if rule_ids:
            self._ask_terms[key] = self._ask_terms.get(key, frozenset()) | frozenset(rule_ids)

    def explicit_hits(self, text: str) -> tuple[PhraseHit, ...]:
        folded = text.casefold()
        hits = []
        for start, end, key in _scan(folded, self._explicit):
            spelling, metrics, values = self._explicit[key]
            hits.append(PhraseHit(start, end, spelling, metrics, values))
        return tuple(hits)

    def uncovered_ambiguous(self, text: str, hits: Sequence[PhraseHit]) -> tuple[tuple[str, str], ...]:
        """(rule id, ambiguous phrase) occurrences no explicit phrase covers."""

        folded = text.casefold()
        found: list[tuple[str, str]] = []
        for key, rule_id in self._ambiguous:
            start = folded.find(key)
            while start >= 0:
                end = start + len(key)
                if not any(hit.start <= start and end <= hit.end for hit in hits):
                    rule = self.catalog.clarification(rule_id)
                    spelling = next(item for item in rule.ambiguous_phrases if item.casefold() == key)
                    found.append((rule_id, spelling))
                start = folded.find(key, start + 1)
        return tuple(found)

    def rules_named_by_ask(self, text: str) -> frozenset[str]:
        """Every rule the ask's wording touches (weak signal): any phrase,
        ambiguous phrase or marker of the rule."""

        folded = text.casefold()
        named: set[str] = set()
        for _, _, key in _scan(folded, self._ask_terms):
            named |= self._ask_terms[key]
        return frozenset(named)

    def ask_signals(self, text: str) -> dict[str, tuple[str, ...]]:
        """The strong signals of the ask's wording per rule: ``two_values``
        (it names two or more of the rule's values) and ``marker`` (one of the
        rule's own ask_markers).  Repeating one value (the question's metric)
        or an ambiguous phrase is not asking about the rule."""

        folded = text.casefold()
        hits = self.explicit_hits(text)
        signals: dict[str, tuple[str, ...]] = {}
        for rule in self.rules:
            kinds = []
            if len(rule.values) > 1 and len(_values_named(hits, rule.id)) >= 2:
                kinds.append("two_values")
            if any(marker.casefold() in folded for marker in rule.ask_markers):
                kinds.append("marker")
            if kinds:
                signals[rule.id] = tuple(kinds)
        return signals

    def rules_asked_by(self, text: str) -> frozenset[str]:
        """Rules the ask's wording really asks the user to choose in (strong signal)."""

        return frozenset(self.ask_signals(text))


def _values_named(hits: Iterable[PhraseHit], rule_id: str) -> frozenset[str]:
    return frozenset(value for hit in hits for owner, value in hit.values if owner == rule_id)


RuleStatus = Literal["single", "resolved", "open", "mixed", "silent"]


@dataclass(frozen=True)
class RuleReading:
    """What a question (plus the user's clarification answers) says about one rule."""

    rule: ClarificationRule
    question_values: frozenset[str]
    answer_values: frozenset[str]
    confirmed_values: frozenset[str]
    ambiguous: tuple[str, ...]

    @property
    def single(self) -> bool:
        return len(self.rule.values) == 1

    @property
    def selected(self) -> frozenset[str]:
        """Values the user chose in a clarification (answer text or confirmed slot)."""

        return self.answer_values | self.confirmed_values

    @property
    def named(self) -> frozenset[str]:
        return self.question_values | self.selected

    @property
    def status(self) -> RuleStatus:
        if self.single:
            return "single"
        if self.selected:
            return "resolved"
        if self.ambiguous and not self.question_values:
            return "open"
        if self.ambiguous:
            return "mixed"
        if self.question_values:
            return "resolved"
        return "silent"


@dataclass(frozen=True)
class ClarificationReading:
    index: PhraseIndex
    question_hits: tuple[PhraseHit, ...]
    answer_hits: tuple[PhraseHit, ...]
    confirmed_metrics: frozenset[str]
    rules: tuple[RuleReading, ...]

    def rule(self, rule_id: str) -> RuleReading | None:
        return next((item for item in self.rules if item.rule.id == rule_id), None)


def read_clarifications(
    catalog: SemanticCatalog,
    question: str,
    answers: Sequence[str] = (),
    confirmed_metrics: Iterable[str] = (),
) -> ClarificationReading:
    """Read the phrase table against the run's question, answers and confirmed metrics."""

    index = PhraseIndex(catalog)
    confirmed = frozenset(metric.removeprefix("metric.") for metric in confirmed_metrics)
    question_hits = index.explicit_hits(question)
    answer_text = "\n".join(answers)
    answer_hits = index.explicit_hits(answer_text)
    ambiguous = index.uncovered_ambiguous(question, question_hits) + index.uncovered_ambiguous(answer_text, answer_hits)
    readings = tuple(
        RuleReading(
            rule=rule,
            question_values=_values_named(question_hits, rule.id),
            answer_values=_values_named(answer_hits, rule.id),
            confirmed_values=frozenset(value.value for value in rule.values if value.metric in confirmed),
            ambiguous=tuple(dict.fromkeys(phrase for rule_id, phrase in ambiguous if rule_id == rule.id)),
        )
        for rule in index.rules
    )
    return ClarificationReading(index, question_hits, answer_hits, confirmed, readings)


# ---------------------------------------------------------------------------
# Checking a model's declared metrics (before any SQL runs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclarationVerdict:
    kind: Literal["clarify", "unsupported", "contradicts"]
    rule: ClarificationRule
    value: ClarificationValue | None = None


def check_declaration(reading: ClarificationReading, metric_ids: Iterable[str]) -> DeclarationVerdict | None:
    """``clarify`` when the model picked a value the wording leaves open;
    ``unsupported`` when the wording (or the user's choice) names an
    unsupported value and none of the declared ones; ``contradicts`` when
    the question names the rule's values, leaves nothing open, the user chose
    nothing, and the model declared only other values of the rule.  Rules are checked in
    catalog order, so one question is asked at a time."""

    declared_metrics = {metric.removeprefix("metric.") for metric in metric_ids}
    for item in reading.rules:
        if item.single:
            continue
        declared = frozenset(value.value for value in item.rule.values if value.metric in declared_metrics)
        if not declared:
            continue
        unsupported = [value for value in item.rule.values if value.value in item.named and not value.supported]
        if unsupported and not declared & item.named:
            return DeclarationVerdict("unsupported", item.rule, unsupported[0])
        if item.selected:
            continue
        if item.ambiguous and not declared <= item.question_values:
            return DeclarationVerdict("clarify", item.rule)
        if item.question_values and not item.ambiguous and not declared & item.question_values:
            # Only a declaration naming none of the question's values: one that
            # also covers them may answer wording the table does not list
            # ("订单总额和净额" declaring gross_fen with net_fen).
            return DeclarationVerdict("contradicts", item.rule)
    return None


# ---------------------------------------------------------------------------
# Reviewing a model's ask_user
# ---------------------------------------------------------------------------


AskSignal = Literal["clarification_id", "two_values", "marker", "weak_only", "none"]
_SIGNAL_ORDER = ("clarification_id", "two_values", "marker")


@dataclass(frozen=True)
class AskVerdict:
    decision: Literal["model_text", "catalog_question", "not_needed"]
    rule: ClarificationRule | None
    targeted: tuple[str, ...]
    id_status: Literal["absent", "declared", "unknown"]
    # The signal the decision rests on, a fixed identifier (never ask text):
    # the decided rule's strongest signal (clarification_id > two_values >
    # marker), else weak_only (the ask touched a rule only weakly) or none.
    signal: AskSignal = "none"



def review_ask(reading: ClarificationReading, clarification_id: str | None, ask_text: str) -> AskVerdict:
    """Which catalog rule the model asks about, and whether asking is warranted.

    A rule the wording leaves open (open/mixed) is asked on its catalog
    question whenever the ask touches it at all.  Sending the model back (a
    rule the wording settles), or replacing its question for a rule the
    question never mentions, needs a strong signal: the declared
    ``clarification_id``, two or more of the rule's values, or one of its
    ask_markers.  So leaving the id out does not skip the check, while an
    ask that only repeats the question's metric (a time-range ask) keeps the
    model's own question.
    """

    known = {item.rule.id for item in reading.rules}
    if clarification_id is None:
        id_status: Literal["absent", "declared", "unknown"] = "absent"
    elif clarification_id in known:
        id_status = "declared"
    else:
        id_status = "unknown"
    signals = {rule_id: set(kinds) for rule_id, kinds in reading.index.ask_signals(ask_text).items()}
    if id_status == "declared":
        signals.setdefault(str(clarification_id), set()).add("clarification_id")
    asked = set(signals)
    touched = asked | reading.index.rules_named_by_ask(ask_text)
    targeted = tuple(item for item in reading.rules if item.rule.id in touched)
    ids = tuple(item.rule.id for item in targeted)

    def signal_of(rule_id: str) -> AskSignal:
        return next((kind for kind in _SIGNAL_ORDER if kind in signals.get(rule_id, ())), "weak_only")

    open_rules = [item for item in targeted if item.status in {"open", "mixed"}]
    if open_rules:
        rule = open_rules[0].rule
        return AskVerdict("catalog_question", rule, ids, id_status, signal_of(rule.id))
    blocking = [item for item in targeted if item.rule.id in asked and item.status in {"resolved", "single"}]
    if blocking:
        rule = blocking[0].rule
        return AskVerdict("not_needed", rule, ids, id_status, signal_of(rule.id))
    silent = [item for item in targeted if item.rule.id in asked and item.status == "silent"]
    if silent:
        rule = silent[0].rule
        return AskVerdict("catalog_question", rule, ids, id_status, signal_of(rule.id))
    return AskVerdict("model_text", None, ids, id_status, "weak_only" if targeted else "none")


def _named_content(reading: ClarificationReading, rule: ClarificationRule) -> dict[str, object]:
    """The rule id, the catalog spellings of its values named in the question
    or answers, and their supported metrics (never model or user text)."""

    item = reading.rule(rule.id)
    named = sorted(item.named) if item is not None else []
    phrases = [
        hit.phrase
        for hit in reading.question_hits + reading.answer_hits
        if any(owner == rule.id for owner, _ in hit.values)
    ]
    metrics = [
        value.metric for value in rule.values if value.value in named and value.metric is not None and value.supported
    ]
    return {
        "clarification_id": rule.id,
        "named_phrases": list(dict.fromkeys(phrases)),
        "declare_metrics": list(dict.fromkeys(metrics)),
    }


def _window_only(window: Mapping[str, str] | None) -> dict[str, str] | None:
    return {"start": window["start"], "end": window["end"]} if window is not None else None


# The B3b text, with or without a request window.  A request without a window
# usually still states the month in the question (every W05 critical case), so
# a hint that also allowed "ask only for the time range" sent a real model back
# to asking about the basis (B3c-1 R2).  The time-range rule stays in the
# contract (CLARIFICATION_APPLY_RULE).
NOT_NEEDED_ACTION = (
    "Do not ask_user about this rule: the question already names its value (or the rule has one value). "
    "Send a tool_call named query_readonly that declares the named metric."
)
CONTRADICTION_ACTION = (
    "Nothing was executed: the question names this rule's value and the declaration picked another one. "
    "Declare the metric the question names (declare_metrics) instead."
)


def not_needed_hint(
    reading: ClarificationReading,
    rule: ClarificationRule,
    request_time_window: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Repair content for an ask the wording settles: fixed text and catalog strings only."""

    return {
        "action": NOT_NEEDED_ACTION,
        **_named_content(reading, rule),
        "request_time_window": _window_only(request_time_window),
    }


def contradiction_hint(
    reading: ClarificationReading,
    rule: ClarificationRule,
    request_time_window: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Repair content for a declaration the question contradicts: fixed text and catalog strings only."""

    return {
        "action": CONTRADICTION_ACTION,
        **_named_content(reading, rule),
        "request_time_window": _window_only(request_time_window),
    }


# ---------------------------------------------------------------------------
# Resume: which rule is waiting, which value the user chose
# ---------------------------------------------------------------------------


def rule_named_by_waiting_question(catalog: SemanticCatalog, question: str) -> ClarificationRule | None:
    """Legacy checkpoints (no clarification id): a question naming two or more
    values of one rule asks about that rule."""

    index = PhraseIndex(catalog)
    hits = index.explicit_hits(question)
    for rule in index.rules:
        if len(rule.values) > 1 and len(_values_named(hits, rule.id)) >= 2:
            return rule
    return None


def select_value(catalog: SemanticCatalog, rule: ClarificationRule, answer: str) -> ClarificationValue | None:
    """The value an answer chooses: exactly one of the rule's values must be named."""

    named = _values_named(PhraseIndex(catalog).explicit_hits(answer), rule.id)
    if len(named) != 1:
        return None
    return rule.value(next(iter(named)))


# ---------------------------------------------------------------------------
# Basis of a verified answer
# ---------------------------------------------------------------------------


def _hits_for_metric(hits: Sequence[PhraseHit], rules: Sequence[ClarificationRule], metric_id: str) -> list[PhraseHit]:
    implied = {(rule.id, value.value) for rule in rules for value in rule.values if value.metric == metric_id}
    return [hit for hit in hits if metric_id in hit.metrics or hit.values & implied]


def metric_basis_note(reading: ClarificationReading, metric_id: str) -> str:
    """One sentence of basis for a verified metric, from catalog strings only."""

    catalog = reading.index.catalog
    name = catalog.metric_name(metric_id)
    head = f"口径：{name}（{metric_id}）"
    rules = reading.index.rules
    in_question = _hits_for_metric(reading.question_hits, rules, metric_id)
    in_answer = _hits_for_metric(reading.answer_hits, rules, metric_id)
    if in_question:
        phrase = max(in_question, key=lambda hit: hit.end - hit.start).phrase
        note = f"{head}；依据：问题中提到‘{phrase}’。"
    elif in_answer:
        phrase = max(in_answer, key=lambda hit: hit.end - hit.start).phrase
        note = f"{head}；依据：你在追问中选择了‘{phrase}’。"
    elif metric_id in reading.confirmed_metrics:
        note = f"{head}；依据：你在追问中选择了‘{name}’。"
    else:
        rule = next(
            (rule for rule in rules if len(rule.values) > 1 and any(value.value == metric_id for value in rule.values)),
            None,
        )
        item = reading.rule(rule.id) if rule is not None else None
        if item is not None and item.named:
            # The question (or the user) names another value of this rule: never
            # claim the wording left the basis open.  The declaration gate stops
            # this before SQL, so this is only a guard.
            note = f"{head}；依据：按目录定义统计。"
        elif rule is not None:
            others = "或".join(
                catalog.metric_name(value.value)
                for value in rule.values
                if value.value != metric_id and value.metric is not None and value.supported
            )
            note = f"{head}；问题中没有写明口径，按{name}统计；如需{others}，请说明。"
        else:
            note = f"{head}；依据：按目录定义统计。"
    for rule in rules:
        if len(rule.values) == 1 and rule.values[0].metric == metric_id and rule.values[0].definition:
            note += f"前提：{rule.values[0].definition}"
    return note


__all__ = [
    "AskVerdict",
    "ClarificationReading",
    "DeclarationVerdict",
    "PhraseHit",
    "PhraseIndex",
    "RuleReading",
    "check_declaration",
    "contradiction_hint",
    "metric_basis_note",
    "not_needed_hint",
    "read_clarifications",
    "review_ask",
    "rule_named_by_waiting_question",
    "select_value",
]
