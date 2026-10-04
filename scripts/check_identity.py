import os
from uuid import uuid4

from queryshield.auth.identity import IDENTITY_CONFIG, resolve_identity


def main() -> int:
    missing = [
        identity["token_env"]
        for identity in IDENTITY_CONFIG.values()
        if not os.getenv(identity["token_env"], "").strip()
    ]
    if missing:
        print("identity_check_blocked reason=identity_config_missing")
        return 2

    for identity_name, expected in IDENTITY_CONFIG.items():
        token = os.environ[expected["token_env"]].strip()
        resolved = resolve_identity(f"Bearer {token}")
        expected_identity = {
            "principal_id": expected["principal_id"],
            "tenant_id": expected["tenant_id"],
            "role": expected["role"],
        }
        assert resolved == expected_identity, identity_name
        print(
            f"{identity_name}=principal:{resolved['principal_id']} "
            f"tenant:{resolved['tenant_id']} role:{resolved['role']}"
        )

    assert resolve_identity(f"Bearer {uuid4()}") is None
    assert resolve_identity("Basic local-probe") is None
    print("identity_check_pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
