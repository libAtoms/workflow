import os
import subprocess

import wfl


def get_wfl_version():
    try:
        with subprocess.Popen("cd '" + os.path.dirname(__file__) + "'; " +
                              "git describe --always --tags --dirty",
                              shell=True, stdout=subprocess.PIPE,
                              env={'PATH': os.environ['PATH']}) as gitv:
            version_str = gitv.stdout.read().strip().decode('utf-8')
    except Exception:
        version_str = ''

    if len(version_str.strip()) == 0:
        try:
            version_str = wfl.__version__
        except AttributeError:
            version_str = 'None'
    else:
        version_str = 'git ' + version_str

    return version_str
