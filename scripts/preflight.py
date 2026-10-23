#!/usr/bin/env python3
"""Read-only checks for local AIRA deployments; uses Python's standard library only.

This is *not* a substitute for authentication or network-layer security.
"""
import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, delimiter, value = line.partition("=")
        if delimiter and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key.strip()):
            # Deliberately do not interpolate shell expansions or evaluate env content.
            values[key.strip()] = value.strip().strip('"\'')
    return values


def audit(config: dict[str, str], compose: str, *, hardened_local: bool) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    secret = config.get("POSTGRES_PASSWORD", "")
    if not secret or secret == "aira_local" or len(secret) < 12:
        (errors if hardened_local else warnings).append(
            "Set a unique POSTGRES_PASSWORD (12+ characters); the Compose fallback is for development only."
        )
    if not all(f'"127.0.0.1:{port}:' in compose for port in ("3000", "8000")):
        errors.append("Expected web and API published ports to be bound to 127.0.0.1 in compose.yml.")
    if config.get("OTEL_CAPTURE_CONTENT", "false").lower() in ("1", "true", "yes"):
        warnings.append("OTEL_CAPTURE_CONTENT enabled: prompts and retrieved content may be stored in traces.")
    warnings.append("AIRA has no user authentication; do not expose it directly to the public Internet.")
    return errors, warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check local AIRA configuration without contacting services.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--hardened-local", action="store_true", help="Fail on development credentials.")
    args = parser.parse_args(argv)
    values = {**read_env(args.env_file), **os.environ}
    compose_path = ROOT / "compose.yml"
    if not compose_path.is_file():
        print("ERROR: compose.yml is missing", file=sys.stderr)
        return 1
    errors, warnings = audit(values, compose_path.read_text(encoding="utf-8"), hardened_local=args.hardened_local)
    for warning in warnings:
        print("WARNING:", warning)
    for error in errors:
        print("ERROR:", error, file=sys.stderr)
    if not errors:
        print("Preflight passed (local-only; not an Internet deployment security audit).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
