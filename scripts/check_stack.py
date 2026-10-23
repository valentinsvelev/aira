#!/usr/bin/env python3
"""Read-only, model-free smoke check against a running local AIRA stack."""
import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def check(base: str, ui: str, timeout: float = 5.0) -> tuple[bool, list[str]]:
    messages = []
    passed = True
    for label, url, expected in (
        ("API liveness", f"{base.rstrip('/')}/health", "ok"),
        ("API/database readiness", f"{base.rstrip('/')}/ready", "ready"),
    ):
        try:
            with urlopen(url, timeout=timeout) as response:
                payload = json.load(response)
                if response.status != 200 or payload.get("status") != expected:
                    raise ValueError("Unexpected health response")
            messages.append(f"PASS: {label}")
        except (HTTPError, URLError, ValueError, TimeoutError, OSError) as exc:
            passed = False
            messages.append(f"FAIL: {label} ({type(exc).__name__})")
    try:
        with urlopen(ui.rstrip("/") + "/", timeout=timeout) as response:
            if response.status != 200:
                raise ValueError("Unexpected web response")
        messages.append("PASS: web application")
    except (HTTPError, URLError, ValueError, TimeoutError, OSError) as exc:
        passed = False
        messages.append(f"FAIL: web application ({type(exc).__name__})")
    return passed, messages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check AIRA readiness; no model calls or tracing.")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--web", default="http://127.0.0.1:3000")
    args = parser.parse_args(argv)
    ok, messages = check(args.api, args.web)
    print("\n".join(messages))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
