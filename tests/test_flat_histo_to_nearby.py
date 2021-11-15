import numpy as np
from ase.atoms import Atoms

from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.select_configs.flat_histogram import biased_select_conf
from wfl.selection_space import val_relative_to_nearby_composition_volume_min


def c(at):
    return (at.get_volume() / len(at), sum(at.numbers == 6) / len(at), sum(at.numbers == 1) / len(at))


def test_flat_histo_to_nearby(tmp_path):
    np.random.seed(5)

    n_at = 30
    n3 = 0

    p_center = (5.5, 0.5, 0.0)

    ats = []
    for at_i in range(1000):
        n1 = np.random.randint(n_at + 1)
        ################################################################################
        # n2 = np.random.randint((n_at-n1) + 1)
        # if n1+n2 < n_at:
        # n3 = np.random.randint((n_at - n1 - n2) + 1)
        # else:
        # n3 = 0
        ################################################################################
        n2 = n_at - n1
        ################################################################################
        at = Atoms(f'Si{n1}C{n2}H{n3}', cell=(1, 1, 1), pbc=[True] * 3)
        at.cell *= (len(at) * np.random.uniform(3, 8)) ** (1.0 / 3.0)

        ################################################################################
        # E_min = np.linalg.norm(np.array(c(at))-p_center)
        ################################################################################
        E_min = np.linalg.norm(np.array(c(at)) - p_center)
        ################################################################################

        at.info['energy'] = np.random.uniform(E_min, 5.0)
        at.info['config_i'] = at_i
        ats.append(at)

    output_ats = val_relative_to_nearby_composition_volume_min(
        ConfigSet_in(input_configs=ats),
        ConfigSet_out(file_root=tmp_path, output_files='test_flat_histo_relative.xyz'),
        1.0, 0.25, 'energy', 'E_dist_to_nearby')

    selected_ats = biased_select_conf(ConfigSet_in(input_configsets=output_ats),
                                      ConfigSet_out(file_root=tmp_path,
                                                    output_files='test_flat_histo_relative.selected.xyz'), 10,
                                      'E_dist_to_nearby', kT=0.1)

    print("selected_ats", [at.info['config_i'] for at in selected_ats])
    assert [at.info['config_i'] for at in selected_ats] == [72, 120, 130, 137, 211, 345, 419, 544, 883, 935]
