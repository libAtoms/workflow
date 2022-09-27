import os
import subprocess
import shlex
from pathlib import Path

def julia_exec_path():
    return os.environ.get("WFL_JULIA_COMMAND", "julia")

def ace_fit_jl_path(julia_exec):
    ace_path = Path(subprocess.check_output(shlex.split(julia_exec), text=True, input="import(ACE1pack)\nprint(pathof(ACE1pack)\n)")).parent.parent
    return str(ace_path / "scripts" / "ace_fit.jl")

