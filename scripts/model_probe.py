from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from queryshield.providers.contracts import ModelProviderError  # noqa: E402
from queryshield.providers.fake_model import FakeModel  # noqa: E402
from queryshield.providers.openai_compatible import (  # noqa: E402
    OpenAICompatibleModel,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a provider-only probe without business data."
    )
    parser.add_argument("-Mode", "--mode", choices=("fake", "real"), default="fake")
    parser.add_argument("-Output", "--output", type=Path)
    args = parser.parse_args()

    messages = [
        {
            "role": "user",
            "content": "Reply with a short provider health response; do not use business data.",
        }
    ]

    if args.mode == "fake":
        result = FakeModel().complete(messages, request_id="probe-request-fake")
        record = _probe_record(
            result.to_redacted_record(), network_request_made=False
        )
        _write_record(args.output, record)
        print(
            "model_probe_ok "
            f"mode=fake model_call_id={record['model_call_id']} "
            f"usage_status={record['usage_status']}"
        )
        return 0

    try:
        result = OpenAICompatibleModel.from_env().complete(messages)
    except ModelProviderError as exc:
        record = _probe_record(
            exc.record,
            network_request_made=exc.record.get("status") == "failed",
        )
        _write_record(args.output, record)
        if exc.code == "missing_model_configuration":
            print("model_probe_blocked reason=missing_model_configuration")
            return 2
        print(f"model_probe_fail reason={exc.code}")
        return 1
    except Exception:
        _write_record(
            args.output,
            _probe_record(
                {
                    "status": "failed",
                    "mode": "real",
                    "provider": "openai-compatible",
                    "model": None,
                    "request_id": None,
                    "model_call_id": None,
                    "provider_request_id": None,
                    "stream": False,
                    "content_present": False,
                    "content_length": 0,
                    "usage": None,
                    "usage_status": "unknown",
                    "error_code": "program_error",
                },
                network_request_made=False,
            ),
        )
        print("model_probe_fail reason=program_error")
        return 1

    record = _probe_record(
        result.to_redacted_record(), network_request_made=True
    )
    _write_record(args.output, record)
    print(
        "model_probe_ok "
        f"mode=real model={record['model']} "
        f"request_id={record['request_id']} "
        f"model_call_id={record['model_call_id']} "
        f"provider_request_id={record['provider_request_id']} "
        f"usage_status={record['usage_status']}"
    )
    return 0


def _write_record(path: Path | None, record: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _probe_record(
    record: dict[str, object], *, network_request_made: bool
) -> dict[str, object]:
    """Add probe-only safety facts without retaining prompt or response text."""
    enriched = dict(record)
    enriched["network_request_made"] = network_request_made
    enriched["business_data_included"] = False
    return enriched


if __name__ == "__main__":
    raise SystemExit(main())
