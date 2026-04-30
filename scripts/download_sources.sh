#!/usr/bin/env bash
# Parallel-clone all source datasets into data_sources/. Fail-fast: any clone
# error aborts the whole script with a non-zero exit code.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
mkdir -p "$ROOT/data_sources"
cd "$ROOT/data_sources"

REPOS=(
  "https://github.com/ncsoft/offsetbias"
  "https://github.com/ScalerLab/JudgeBench"
  "https://github.com/lm-sys/FastChat"
  "https://github.com/PKU-Alignment/beavertails"
)

pids=()
for url in "${REPOS[@]}"; do
  name="$(basename "$url")"
  if [[ -d "$name" ]]; then
    echo "[skip] $name already cloned"
    continue
  fi
  echo "[clone] $url"
  git clone --depth 1 "$url" &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

if [[ $fail -ne 0 ]]; then
  echo "ERROR: one or more clones failed" >&2
  exit 1
fi

echo "All sources downloaded into $ROOT/data_sources/"
echo "Note: SummEval and DSTC11 are pulled from HuggingFace at fixture-build time (cached in ~/.cache/huggingface)."
