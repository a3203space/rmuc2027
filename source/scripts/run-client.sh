#!/bin/sh
set -eu
RMUC_PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RMUC_PROJECT_ROOT"
if [ ! -x "$RMUC_PROJECT_ROOT/build/client/bin/RMUC2027Client" ]; then
    echo "客户端尚未构建：请先安装 Qt 6，再执行 cmake --preset release 和 cmake --build --preset release。" >&2
    echo "Ubuntu 22.04 依赖安装命令见 README.md 的“Ubuntu 22.04 安装 Qt”章节。" >&2
    exit 1
fi
exec "$RMUC_PROJECT_ROOT/build/client/bin/RMUC2027Client" "$@"
