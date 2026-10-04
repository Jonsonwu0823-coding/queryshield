import importlib


REQUIRED_MODULES = ("fastapi", "pydantic", "psycopg", "uvicorn", "httpx")


def main() -> int:
    missing = []
    probe_errors = []

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing.append(f"{module_name}: {exc}")
        except ImportError as exc:
            probe_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            probe_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    if missing:
        print("runtime_dependencies_missing")
        for item in missing:
            print(item)
        return 2

    if probe_errors:
        print("runtime_dependency_probe_failed")
        for item in probe_errors:
            print(item)
        return 1

    print("runtime_dependencies_ok=" + ",".join(REQUIRED_MODULES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
