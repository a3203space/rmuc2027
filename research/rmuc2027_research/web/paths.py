"""Workspace resources and isolated web application state."""
from pathlib import Path
from rmuc2027_core.paths import ROOT, data_path as _data_path

def data_path(name):
    return _data_path("research/web") / name

STRATEGIES = ROOT / "packages/core/bots/catalog"
