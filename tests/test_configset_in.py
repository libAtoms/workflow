from pathlib import Path

import pytest
from ase.atoms import Atoms
import ase.io

from wfl.configset import ConfigSet_in, ConfigSet_out

def test_configset_from_atoms():
    ats = [Atoms('H'), Atoms('C')]
    c_ats = ConfigSet_in(input_configs=ats)

    assert len([at for at in c_ats]) == 2


def test_configset_from_files(tmp_path):
    ats = [Atoms('H'), Atoms('C')]
    ase.io.write(tmp_path / 'at1.xyz', ats)
    ase.io.write(tmp_path / 'at2.xyz', ats)

    c_ats = ConfigSet_in(input_files=[str(tmp_path / 'at1.xyz'), str(tmp_path / 'at2.xyz')])

    assert len([at for at in c_ats]) == 4


def test_configset_from_ConfigSets(tmp_path):
    ats = [Atoms('H'), Atoms('C')]
    ase.io.write(tmp_path / 'at1.xyz', ats)
    ase.io.write(tmp_path / 'at2.xyz', ats)

    c_ats = ConfigSet_in(input_configsets=[ConfigSet_in(input_files=str(tmp_path / 'at1.xyz')), ConfigSet_in(input_files=str(tmp_path / 'at2.xyz'))])

    assert len([at for at in c_ats]) == 4


@pytest.mark.xfail()
def test_configset_from_ConfigSets_mismatched(tmp_path):
    ats = [Atoms('H'), Atoms('C')]
    ase.io.write(tmp_path / 'at1.xyz', ats)

    c_ats = ConfigSet_in(input_configsets=[ConfigSet_in(input_files=str(tmp_path / 'at1.xyz')), ConfigSet_in(input_configs=ats)])


def test_merge():
    ats1 = [Atoms('H'), Atoms('C')]
    c_ats1 = ConfigSet_in(input_configs=ats1)

    ats2 = [Atoms('B'), Atoms('F')]
    c_ats2 = ConfigSet_in(input_configs=ats2)

    c_ats1.merge(c_ats2)

    # merge with additional content
    assert len([at for at in c_ats1]) == 4

    # merge with empty
    c_ats1.merge(ConfigSet_in())

    assert len([at for at in c_ats1]) == 4

def test_scratch(tmp_path):
    ats = [Atoms('H'), Atoms('C')]
    ase.io.write(tmp_path / 'at1.xyz', ats)
    ase.io.write(tmp_path / 'at2.xyz', ats)

    ci = ConfigSet_in(input_files=[str(tmp_path / 'at1.xyz'), str(tmp_path / 'at2.xyz')])

    ci_s = ci.to_file(tmp_path / 'combined.xyz')
    assert Path(ci_s).name == 'combined.xyz'
    assert len(ase.io.read(ci_s, ':')) == 4

    ci_s = ci.to_file(tmp_path / 'combined.xyz', scratch=True)
    assert (Path(ci_s).name.startswith('combined.') and
            Path(ci_s).name.endswith('.xyz') and
            Path(ci_s).name != 'combined.xyz')
    assert len(ase.io.read(ci_s, ':')) == 4
