import os

def julia_exec_path():
    return os.environ.get("WFL_JULIA_COMMAND", "julia")
