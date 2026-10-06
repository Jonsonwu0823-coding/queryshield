"""The one rule for ``query_readonly`` parameter objects keyed by position."""

from __future__ import annotations

from collections.abc import Mapping


def ordered_param_values(params: Mapping[object, object]) -> tuple[object, ...] | None:
    """Return the values in index order, or None unless the keys are exactly "0".."n-1".

    Keys must be the canonical decimal strings, so ``"01"`` and ``"٠"`` are rejected;
    the keys are compared as strings, never parsed, so an enormous digit string cannot raise.
    """

    if any(type(key) is not str for key in params) or set(params) != {str(index) for index in range(len(params))}:
        return None
    return tuple(params[str(index)] for index in range(len(params)))
