import os


IDENTITY_CONFIG: dict[str, dict[str, str]] = {
    "a-requester": {
        "token_env": "QUERYSHIELD_TOKEN_A_REQUESTER",
        "principal_id": "a-requester",
        "tenant_id": "A",
        "role": "requester",
    },
    "a-approver": {
        "token_env": "QUERYSHIELD_TOKEN_A_APPROVER",
        "principal_id": "a-approver",
        "tenant_id": "A",
        "role": "approver",
    },
    "b-requester": {
        "token_env": "QUERYSHIELD_TOKEN_B_REQUESTER",
        "principal_id": "b-requester",
        "tenant_id": "B",
        "role": "requester",
    },
    "b-approver": {
        "token_env": "QUERYSHIELD_TOKEN_B_APPROVER",
        "principal_id": "b-approver",
        "tenant_id": "B",
        "role": "approver",
    },
}


def _configured_tokens() -> dict[str, dict[str, str]]:
    token_map: dict[str, dict[str, str]] = {}

    for identity in IDENTITY_CONFIG.values():
        token = os.getenv(identity["token_env"], "").strip()
        if not token:
            continue

        if token in token_map:
            return {}

        token_map[token] = {
            "principal_id": identity["principal_id"],
            "tenant_id": identity["tenant_id"],
            "role": identity["role"],
        }

    return token_map


def resolve_identity(
    authorization: str | None,
) -> dict[str, str] | None:
    if authorization is None:
        return None

    parts = authorization.strip().split(maxsplit=1)

    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    token = parts[1].strip()

    if not token:
        return None

    return _configured_tokens().get(token)
