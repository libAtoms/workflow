from ase.atoms import Atoms

import wfl.select.simple as simple
from wfl.configset import ConfigSet, OutputSpec


def test_select_lambda(tmp_path):
    ats = [Atoms('Si', cell=(1, 1, 1), pbc=[True] * 3) for _ in range(40)]
    for at_i, at in enumerate(ats):
        at.info['index'] = at_i

    ci = ConfigSet(ats)
    co = OutputSpec('test_simple.info_in.xyz', file_root=tmp_path)
    selected_ats = simple.by_bool_func(ci, co, lambda at : at.info['index'] in list(range(10, 20)))

    assert len(list(selected_ats)) == 20 - 10
    for at in selected_ats:
        assert at.info['index'] in range(10, 20)


def _pytest_select(at):
    return at.info['index'] in list(range(10, 20))


def test_select_real_func(tmp_path):
    ats = [Atoms('Si', cell=(1, 1, 1), pbc=[True] * 3) for _ in range(40)]
    for at_i, at in enumerate(ats):
        at.info['index'] = at_i

    ci = ConfigSet(ats)
    co = OutputSpec('test_simple.info_in.xyz', file_root=tmp_path)
    selected_ats = simple.by_bool_func(ci, co, _pytest_select)

    assert len(list(selected_ats)) == 20 - 10
    for at in selected_ats:
        assert at.info['index'] in range(10, 20)


def test_by_index(tmp_path):
    ats = [Atoms('Si', cell=(1, 1, 1), pbc=[True] * 3) for _ in range(40)]
    for at_i, at in enumerate(ats):
        at.info['index'] = at_i

    ci = ConfigSet(ats)
    co = OutputSpec('test_simple.indices.xyz', file_root=tmp_path)
    indices = [4, 0, 7, 12, 12, 25, 45, 45]
    selected_ats = simple.by_index(ci, co, indices)

    assert len(list(selected_ats)) == len([i for i in indices if (i >= 0 and i < len(ats))])
    for at in selected_ats:
        assert at.info['index'] in indices
