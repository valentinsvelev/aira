#!/bin/sh
# Development-only bootstrap. Keeps the API available even if optional
# observability packages cannot be downloaded; tracing status reports why.
set -eu

if [ "${OTEL_TRACES_ENABLED:-false}" = "true" ]; then
  if ! python -c 'from opentelemetry.sdk.trace import TracerProvider; from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter' >/dev/null 2>&1; then
    echo "[tracing] Missing OpenTelemetry SDK/exporter; installing small tracing extras into the persistent dev volume..."
    if python -m pip install --disable-pip-version-check --no-input --upgrade \
        --target /opt/aira-tracing-deps -r /app/requirements-observability.txt; then
      echo "[tracing] Optional packages installed."
    else
      echo "[tracing] WARNING: Could not install tracing packages. API will start; check /tracing/status." >&2
    fi
  fi
fi

exec uvicorn aira.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app/src
