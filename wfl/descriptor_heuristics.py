import os
from pathlib import Path

from wfl.utils.replace_eval_in_strs import replace_eval_in_strs


def len_scale_pair(length_scales, t, Z1, Z2):
    """Returns length scale for a pair of elements

    Parameters
    ----------
    length_scales: dict
        dict with Z as keys
    t: str
        type of length scale, key in length_scales[Z1] and Z2
    Z1, Z2: int
        atomic numbers of 2 atoms involved, keys for length_scales

    Returns
    -------
    int
        Z1-Z2 length scale
    """
    return 0.5 * (length_scales[Z1][t][0] + length_scales[Z2][t][0])


def descriptors_from_length_scales(descriptors, Zs, length_scales, SOAP_hypers=None):
    """Create all descriptors needed for all species in system

    Parameters
    ----------
    descriptors: dict or list(dict)
        templates of descriptors to have templated contents replaced and optionally (depending
        on value of add_species) duplicated to account for all species.
    Zs: list(int)
        list of atomic numbers present in the system
    length_scales: dict
        length scales for each species.  Keys are atomic numbers.  For each atomic number has key 'bond_len',
        which points to tuple/list with first element being bond length
    SOAP_hypers: dict, default None
        SOAP hypers from universal SOAP heuristics.  Dict with atomic numbers as keys, values are
        dicts with keys 'cutoff', 'cutoff_transition_width', and 'atom_gaussian_width'.

    Returns
    -------
    """

    if isinstance(descriptors, dict):
        use_descriptors = [descriptors]
    else:
        use_descriptors = descriptors

    repl_dict = {'BOND_LEN_MAX': max([length_scales[Z]['bond_len'][0] for Z in Zs])}

    dup = False

    result_descs = []
    desc_Zs = []
    for descriptor in use_descriptors:
        add_species = descriptor.pop('add_species', 'auto')
        if add_species == 'auto':
            # let species behave as default
            result_descs.append(replace_eval_in_strs(descriptor, repl_dict, n_float_sig_figs=2))
            desc_Zs.append(None)
        elif add_species == 'manual_Z_pair':
            dup = True
            # duplicate descriptors for each Z1-Z2 pair
            for Z1 in Zs:
                for Z2 in Zs:
                    if Z1 > Z2:
                        continue
                    repl_dict = {'BOND_LEN_Z1_Z2': len_scale_pair(length_scales, 'bond_len', Z1, Z2),
                                 'Z1': Z1, 'Z2': Z2}
                    result_descs.append(replace_eval_in_strs(descriptor, repl_dict, n_float_sig_figs=2))
                    result_descs[-1]['add_species'] = False
                    desc_Zs.append((Z1, Z2))

        elif add_species == 'manual_Zcenter':
            dup = True
            # duplicate descriptor for each Zcenter
            for Zcenter in Zs:
                repl_dict = {'BOND_LEN_Z': length_scales[Zcenter]['bond_len'][0],
                             'BOND_LEN_Z_MAX': max(
                                 [len_scale_pair(length_scales, 'bond_len', Zcenter, Z2) for Z2 in Zs]),
                             'Zcenter': Zcenter, 'nZ': len(Zs), 'Zs': list(Zs)}
                result_descs.append(replace_eval_in_strs(descriptor, repl_dict, n_float_sig_figs=2))
                result_descs[-1]['add_species'] = False
                desc_Zs.append(Zcenter)

        elif add_species == 'manual_universal_SOAP':
            dup = True
            if SOAP_hypers is None:
                raise RuntimeError('Got manual_universal_SOAP but not SOAP_hypers')
            # duplicate descriptor for each Zcenter using universal SOAP hypers
            for Zcenter in Zs:
                for h_dict in SOAP_hypers[Zcenter]:
                    repl_dict = {'R_CUT': h_dict['cutoff'], 'R_TRANS': h_dict['cutoff_transition_width'],
                                 'ATOM_SIGMA': h_dict['atom_gaussian_width'],
                                 'Zcenter': Zcenter, 'nZ': len(Zs), 'Zs': list(Zs)}
                    result_descs.append(replace_eval_in_strs(descriptor, repl_dict, n_float_sig_figs=2))
                    result_descs[-1]['add_species'] = False
                    desc_Zs.append(Zcenter)
        elif not add_species:
            # let species behave as default
            result_descs.append(replace_eval_in_strs(descriptor, repl_dict, n_float_sig_figs=2))
            result_descs[-1]['add_species'] = False
            desc_Zs.append(result_descs[-1].get('Zs', None))
        else:
            raise ValueError('Unknown \'add_species\' value \'{}\''.format(add_species))

    if isinstance(descriptors, dict) and not dup:
        return result_descs[0], None
    else:
        return result_descs, desc_Zs


def descriptor_2brn_uniform_file(descriptor, ident='', desc_i=0):
    """Write uniform-in-deformed-space sparse points file for 2-body polynomial descriptors

    **UNTESTED!!!!**

    Parameters
    ----------
    descriptor: dict
        nested structure with some contained dicts that have 'sparse_method' = '\_2BRN_UNIFORM_FILE\_',
        and also ``n_sparse``, ``exponents``, ``cutoff``
    ident: str, default ''
        identifier string to add to sparsepoints filename
    desc_i: int, default 0
        offset to descriptor number in filename

    Returns
    -------
    desc_i: int
        newly incremented desc_i
    """
    if isinstance(descriptor, dict):
        if descriptor.get('sparse_method') == '_2BRN_UNIFORM_FILE_':
            n_sparse = descriptor['n_sparse']
            exponents = descriptor['exponents']
            cutoff = descriptor['cutoff']
            cutoff_transition_width = descriptor.get('cutoff', 0.0)

            sparse_pts = [(x + 1) * (cutoff - cutoff_transition_width) / n_sparse for x in range(n_sparse)]

            sparsepoints_filename = f'input_sparsepoints{ident}_desc_{desc_i}'
            with open(sparsepoints_filename, 'w') as fsp:
                for sparse_pt in sparse_pts:
                    fsp.write('1.0\n')
                    for exponent in exponents:
                        fsp.write('{}\n'.format(sparse_pt ** exponent))

            descriptor['sparse_method'] = 'file'
            descriptor['sparse_file'] = str(Path(os.getcwd()) / sparsepoints_filename)

            desc_i += 1
        else:
            for k in descriptor:
                desc_i = descriptor_2brn_uniform_file(descriptor[k], desc_i=desc_i)
    elif isinstance(descriptor, list):
        for v in descriptor:
            desc_i = descriptor_2brn_uniform_file(v, desc_i=desc_i)

    return desc_i
