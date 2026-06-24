#!/usr/bin/env bash

set -euo pipefail

echo "[INFO] Stopping existing Ray processes..."

ray stop --force >/dev/null 2>&1 || true
sleep 3

leftovers="$(
  pgrep -af \
    'raylet|gcs_server|dashboard_agent|ActorGroup|RolloutGroup|run_embodiment' \
    || true
)"

if [[ -n "$leftovers" ]]; then
  echo "[WARN] Possible remaining processes:"
  echo "$leftovers"
else
  echo "[OK] No relevant Ray or RLinf worker processes remain."
fi
