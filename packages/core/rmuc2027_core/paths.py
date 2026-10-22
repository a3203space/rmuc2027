"""Resource root and writable user data, shared by source and installed runtimes."""
import os
from pathlib import Path
ROOT = Path(os.environ.get('RMUC_RESOURCE_ROOT', Path(__file__).resolve().parents[3])).resolve()
def data_path(name):
    return Path(os.environ.get('RMUC_USER_DATA', ROOT/'data')).resolve()/name
