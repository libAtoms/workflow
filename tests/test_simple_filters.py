from ase.atoms import Atoms

import wfl.select_configs.simple_filters as simple_filters
from wfl.configset import ConfigSet_in, ConfigSet_out


def test_all_in(tmp_path):
    ats = [Atoms('Si', cell=(1, 1, 1), pbc=[True] * 3) for _ in range(40)]
    for at_i, at in enumerate(ats):
        at.info['index'] = at_i

    ci = ConfigSet_in(input_configs=ats)
    co = ConfigSet_out(file_root=tmp_path, output_files='test_simple_filters.info_in.xyz')
    selected_ats = simple_filters.apply(ci, co, simple_filters.InfoAllIn(('index', range(10, 20))))

    assert len(list(selected_ats)) == 20 - 10
    for at in selected_ats:
        assert at.info['index'] in range(10, 20)


def test_all_in2(tmp_path):
    ats = [Atoms('Si', cell=(1, 1, 1), pbc=[True] * 3) for _ in range(40)]
    for at_i, at in enumerate(ats):
        at.info['index'] = at_i
        at.info['tag'] = f'val_{at_i}'

    ci = ConfigSet_in(input_configs=ats)
    co = ConfigSet_out(file_root=tmp_path, output_files='test_simple_filters.info_starts.xyz')
    selected_ats = simple_filters.apply(ci, co, simple_filters.InfoAllStartWith(('tag', 'val_1')))

    assert len(list(selected_ats)) == 11  # 1 or 10-19
    for at in selected_ats:
        assert at.info['index'] in [1] + list(range(10, 20))


def test_by_index(tmp_path):
    ats = [Atoms('Si', cell=(1, 1, 1), pbc=[True] * 3) for _ in range(40)]
    for at_i, at in enumerate(ats):
        at.info['index'] = at_i

    ci = ConfigSet_in(input_configs=ats)
    co = ConfigSet_out(file_root=tmp_path, output_files='test_simple_filters.indices.xyz')
    indices = [4, 0, 7, 12, 12, 25, 45]
    selected_ats = simple_filters.by_index(ci, co, indices)

    assert len(list(selected_ats)) == len([i for i in indices if (i >= 0 and i < len(ats))])
    for at in selected_ats:
        assert at.info['index'] in indices
