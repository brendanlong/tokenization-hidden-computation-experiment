#!/usr/bin/env bash
# Recover rollouts.jsonl from a captured `sky logs` stream, using the
# gzip+base64 escape hatch emitted by skypilot/train-retok-rl.yaml
# (TASK=reversal). Usage:
#
#   scripts/extract_rollouts_from_log.sh stream.log > rollouts.jsonl
#
# Two transforms are load-bearing, in this order:
#   1. strip ANSI colour codes — `sky logs` wraps its per-line prefix in
#      \e[36m...\e[0m, so a prefix-stripping regex anchored at ^( never
#      matches on the raw log;
#   2. strip the "(cluster, pid=N) " prefix itself.
# Verified to round-trip byte-identically against the Run 5 artifact.
set -euo pipefail

awk '/ROLLOUTS_B64_BEGIN/{f=1;next} /ROLLOUTS_B64_END/{f=0} f' "${1:?usage: $0 <sky-logs-capture>}" \
  | sed -E $'s/\x1b\\[[0-9;]*m//g; s/^\\([^)]*\\) //' \
  | tr -d ' \r' \
  | base64 -d \
  | gunzip
