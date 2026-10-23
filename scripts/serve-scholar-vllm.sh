#!/usr/bin/env bash
# Starts ONE optional ScholarQA model on a Linux CUDA machine with vLLM installed.
set -euo pipefail
name="${1:-}"
case "$name" in
  astabrief) model="allenai/AstaBrief_8B"; port="${SCHOLAR_ASTA_PORT:-8002}" ;;
  dr-tulu) model="rl-research/DR-Tulu-8B"; port="${SCHOLAR_DR_PORT:-8001}" ;;
  *) echo "Usage: $0 astabrief|dr-tulu" >&2; exit 2 ;;
esac
exec vllm serve "$model" --host 0.0.0.0 --port "$port" --dtype auto --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}"
