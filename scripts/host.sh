#!/bin/sh
set -eu
RMUC_PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RMUC_PROJECT_ROOT"
exec uv run --locked rmuc2027 host "$@"
