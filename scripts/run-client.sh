#!/bin/sh
set -eu
RMUC_PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RMUC_PROJECT_ROOT"
exec "$RMUC_PROJECT_ROOT/build/client/bin/RMUC2027Client" "$@"
