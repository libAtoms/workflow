"""configuration for expyre, from "config.json" in $EXPYRE_ROOT or $HOME/.expyre

root: str
    path of root directory for this run
systems: dict
    dict of expyre.system.System that jobs can run on
db: JobsDB
    expyre.jobsdb.JobsDB database of jobs
"""
import sys
import os
import json
from pathlib import Path

# read config.json file from expyre root (~/.expyre or EXPYRE_ROOT),
# and save important parts (root, systems, db) as symbols in this module


def _rundir_extra(root):
    if root.name == '.expyre' or root.name == '_expyre':
        use_root = root.parent
    else:
        use_root = root
    return os.environ.get('HOSTNAME', 'unkownhost') + '-' + str(use_root).replace('/', '_')


def _get_config(root_dir):
    if root_dir is not None:
        root = Path(root_dir)
    else:
        # get global settings from $EXPYRE_ROOT or $HOME/.expyre
        root = Path(os.environ.get('EXPYRE_ROOT', Path.home() / '.expyre'))

    try:
        with open(root / 'config.json') as fin:
            config_data = json.load(fin)
        found_config = True
    except FileNotFoundError:
        config_data = {'systems': {}}
        found_config = False

    if root_dir is not None or 'EXPYRE_ROOT' in os.environ:
        # passed in or from env var, so no local overrides
        return root, _rundir_extra(root), config_data

    systems = config_data['systems']

    # look for local _expyre directories that override these defaults, starting in
    # $CWD and going up to $HOME (or /)
    cur_dir = Path.cwd()
    while cur_dir.absolute() != Path.home().absolute():
        if (cur_dir / '_expyre').exists():
            root = cur_dir / '_expyre'

            try:
                with open(cur_dir / '_expyre' / 'config.json') as fin:
                    config_data_loc = json.load(fin)

                assert len(config_data_loc) == 0 or set(config_data_loc.keys()) == {'systems'}

                if 'systems' in config_data_loc:
                    for sys_name, sys_data in config_data_loc['systems'].items():
                        if sys_data is None:
                            # delete
                            try:
                                del systems[sys_name]
                            except KeyError:
                                pass
                        elif sys_name in systems:
                            # update
                            systems[sys_name].update(sys_data)
                        else:
                            # add
                            assert isinstance(sys_data, dict)
                            systems[sys_name] = sys_data

                found_config = True
                break

            except FileNotFoundError:
                pass

        if cur_dir.parent == cur_dir:
            # reached root, do not try to go further up
            break
        else:
            cur_dir = cur_dir.parent

    if not found_config:
        raise FileNotFoundError('Failed to find any config.json file')

    return root, _rundir_extra(root), config_data


root = None
systems = None
db = None


def init(root_dir=None, verbose=False):
    import os

    from .units import time_to_sec, mem_to_kB
    from .system import System
    from .jobsdb import JobsDB

    global root, systems, db

    root, _rundir_extra, _config_data = _get_config(root_dir)

    systems = {}
    for _sys_name in _config_data['systems']:
        _sys_data = _config_data['systems'][_sys_name]
        if _sys_data['partitions'] is not None:
            for _partitions in _sys_data['partitions']:
                _sys_data['partitions'][_partitions]['max_time'] = time_to_sec(_sys_data['partitions'][_partitions]['max_time'])
                _sys_data['partitions'][_partitions]['max_mem'] = mem_to_kB(_sys_data['partitions'][_partitions]['max_mem'])
        systems[_sys_name] = System(rundir_extra=_rundir_extra, **_sys_data)

    db = JobsDB(root / 'jobs.db')
    if verbose:
        sys.stderr.write(f'expyre config got systems {list(systems.keys())}\n')


if 'pytest' not in sys.modules:
    init()
