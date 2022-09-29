import os
import subprocess
import shlex
from pathlib import Path

def julia_exec_path():
    return os.environ.get("WFL_JULIA_COMMAND", "julia")
