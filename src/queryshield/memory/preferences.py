"""Small explicit PreferenceStore; it is not an automatic memory writer."""

from __future__ import annotations

from collections.abc import Mapping

from queryshield.db.w04_state import StateStore


PREFERENCE_KEYS = frozenset({"display_language", "answer_style"})
ALLOWED_PREFERENCE_VALUES = {
    "display_language": frozenset({"zh-CN", "en"}),
    "answer_style": frozenset({"concise", "table"}),
}


class PreferenceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class PreferenceStore:
    def __init__(self, state: StateStore) -> None:
        self.state = state

    def _validate_key(self, key: str) -> None:
        if key not in PREFERENCE_KEYS:
            raise PreferenceError("unknown_preference", "preference key is not supported")

    def get(self, *, tenant_id: str, principal_id: str, key: str) -> dict[str, object] | None:
        self._validate_key(key)
        return self.state.get_preference(tenant_id, principal_id, key)

    def put(self, *, tenant_id: str, principal_id: str, key: str, value: object, confirmed: object) -> dict[str, object]:
        self._validate_key(key)
        if type(confirmed) is not bool or confirmed is not True:
            raise PreferenceError("confirmation_required", "confirmed must be the boolean true")
        if type(value) is not str or value not in ALLOWED_PREFERENCE_VALUES[key]:
            raise PreferenceError("invalid_preference_value", "value is not allowed for this key")
        return self.state.put_preference(
            tenant_id=tenant_id,
            principal_id=principal_id,
            key=key,
            value=value,
        )

    def delete(self, *, tenant_id: str, principal_id: str, key: str) -> None:
        self._validate_key(key)
        self.state.delete_preference(tenant_id, principal_id, key)

    def apply_to_request(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        key: str,
        explicit_value: str | None,
    ) -> str | None:
        """An explicit current request beats a stored confirmed preference."""

        self._validate_key(key)
        if explicit_value is not None:
            return explicit_value
        record = self.get(tenant_id=tenant_id, principal_id=principal_id, key=key)
        return str(record["value"]) if record is not None else None


__all__ = [
    "ALLOWED_PREFERENCE_VALUES",
    "PREFERENCE_KEYS",
    "PreferenceError",
    "PreferenceStore",
]
