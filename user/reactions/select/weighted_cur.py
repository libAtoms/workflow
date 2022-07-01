import numpy as np
try:
    from quippy.descriptors import Descriptor
except ModuleNotFoundError:
    pass

from wfl.configset import ConfigSet, OutputSpec
from wfl.reactions_processing import trajectory_processing

try:
    from asaplib.compressor.cur import cur_column_select
except ModuleNotFoundError:
    cur_column_select = None


def calc_and_choose(frames, num, desc, structure_weights=None, label=None):
    if structure_weights is None:
        structure_weights = np.ones_like(frames)

    vec = desc.calc_descriptor(frames)
    indices = []
    env_weights = []

    for i, v in enumerate(vec):
        if 0 not in np.shape(v):
            indices.extend([i] * len(v))
            env_weights.extend([structure_weights[i]] * len(v))

    indices = np.array(indices)
    vec = np.concatenate([v for v in vec if 0 not in np.shape(v)])

    assert vec.shape[0] == len(indices)

    if label is None:
        label = ""

    print(f"descriptor for CUR for {label:8} of shape {f'{vec.shape}':12} - choosing {num:4}")

    k_nn = np.dot(vec, vec.T) ** 2

    return indices[cur_column_select(k_nn, num)]


def selection(configs_in, configs_out, z_list, descriptor_stubs, limit, n_select):
    """

    Parameters
    ----------
    configs_in: ConfigSet
    configs_out : OutputSpec
    z_list
    descriptor_stubs
    limit
    n_select

    Returns
    -------

    """
    weights = trajectory_processing.calc_structure_weights(configs_in, limit)

    print("weight shape is:", weights.shape)

    chosen_indices = []
    for z in z_list:
        for idesc, descriptor_base in enumerate(descriptor_stubs):
            soap_desc = Descriptor(f"{descriptor_base} Z={z}")
            chosen_indices.append(calc_and_choose(configs_in, n_select[z], soap_desc, weights,
                                                  label=f"{z} descriptor:{idesc}"))

    chosen_indices = np.concatenate(chosen_indices)
    chosen_indices = np.unique(chosen_indices)

    for i, at in enumerate(configs_in):
        if i not in chosen_indices:
            continue
        configs_out.write(at, configs_in.get_current_input_file())


def selection_full_desc(configs_in, configs_out, descriptor_strs, n_select, limit):
    """Just as select, but full descriptor strings need to be given

    Parameters
    ----------
    configs_in: ConfigSet
    configs_out : OutputSpec
    descriptor_strs : dict
        z -> list(descriptor_strins)
    n_select: dict
        z -> num_to_select per descriptor
    limit

    Returns
    -------

    """
    if configs_out.is_done():
        print('output is done, returning')
        return configs_out.to_ConfigSet()

    weights = trajectory_processing.calc_structure_weights(configs_in, limit)

    print("weight shape is:", weights.shape)

    chosen_indices = []

    for z, desc_list in descriptor_strs.items():
        for idesc, desc in enumerate(desc_list):
            soap_desc = Descriptor(desc)
            chosen_indices.append(calc_and_choose(configs_in, n_select[z], soap_desc, weights,
                                                  label=f"{z} descriptor:{idesc}"))

    chosen_indices = np.concatenate(chosen_indices)
    chosen_indices = np.unique(chosen_indices)

    configs_out.pre_write()
    for i, at in enumerate(configs_in):
        if i not in chosen_indices:
            continue
        configs_out.write(at, configs_in.get_current_input_file())
    configs_out.end_write()
