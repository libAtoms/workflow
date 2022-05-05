import re
from copy import deepcopy

import numpy as np
from ase.atoms import Atoms
try:
    from quippy.descriptors import Descriptor
except ModuleNotFoundError:
    pass

from wfl.autoparallelize import autoparallelize
from wfl.utils.quip_cli_strings import dict_to_quip_str


def from_any_to_Descriptor(descriptor_src, verbose=False):
    """Create quippy.descriptors.Descriptor objects

    Parameters
    ----------
    descriptor_src: desc / list(desc) / dict(Zcenter : desc) / dict(Zcenter : list(desc))
        String to pass to quippy Descriptor. "Zcenter" denotes atomic number of the central atom".
        Any "desc" value can be a string, a dict (converted to 
        key=value string), or quippy.descriptors.Descriptor object. 

        Each ``descriptor_src`` type correponds to 
        
            * a dictionary with None as keys: one or more descriptor for for all species.
            * a dictionary with integers as keys: one or more descriptor for for each species.
            * string, dictionary or ``Descriptor``: single descriptor
            * list: list of descriptors to concatenate

    verbose: bool, default False
        verbose output

    Returns
    -------
    descs: dict
        dict of Descriptors objects
    """

    # if not a dict with all keys as None or int, put into a dict with None as key
    if (isinstance(descriptor_src, dict) and all([k is None or isinstance(k, int) for k in descriptor_src])):
        descriptor_src = deepcopy(descriptor_src)
    else:
        descriptor_src = {None: deepcopy(descriptor_src)}

    # enclose bare descriptors in list
    for Zcenter, d in descriptor_src.items():
        if isinstance(d, (str, dict, Descriptor)):
            descriptor_src[Zcenter] = [d]

    # convert from dicts or strings to Descriptors
    descs = {}
    for Zcenter, d_list in descriptor_src.items():
        descs[Zcenter] = []
        for d in descriptor_src[Zcenter]:
            if verbose:
                print('Zcenter', Zcenter, 'desc', d)

            if isinstance(d, dict):
                d = dict_to_quip_str(d)

            if isinstance(d, str):
                d = Descriptor(d)

            if not isinstance(d, Descriptor):
                raise RuntimeError(f'Got descriptor of unknown type {type(d)}, not dict or str or Descriptor')

            descs[Zcenter].append(d)

    return descs


def calc(inputs, outputs, descs, key, local=False, normalize=True, composition_weight=True, force=False, verbose=False):
    """Calculates descriptors on a set of configs, I/O with ConfigSet_{in,out}

    Parameters
    ----------
    inputs: ConfigSet
        input configurations
    outputs: OutputSpec
        where to write outputs
    descs: str / list(str) / dict(Z : str)
        descriptor (string or dict or quippy.descriptors.Descriptor) or list of descriptors (applied to all species) 
        or dict of descriptors for each species (key is species or None for all species)
        If global, combined descriptor will be concatenated.  If local and Z is not None, multiple arrays
        entries will be created, one per Zcenter, named <key>_Z_<Zcenter>.
    key: str
        info/arrays key to store descriptor vectors
    local: bool, default False
        calculate descriptors for each atom
    normalize: bool, default True
        normalize final vector (e.g. if contributions from multiple species for a global were concatenated)
    composition_weight: bool, default True
        when concatenating contributions from different species for a global, weight each by composition fraction
    force: bool, default False
        overwrite key if already exists
    verbose: bool, default False
        verbose output

    Returns
    -------
    ConfigSet 
        OutputSpec.to_ConfigSet() Pointing to outputs
    """
    return autoparallelize(iterable=inputs, outputspec=outputs, op=calc_autopara_wrappable, descs=descs, key=key, local=local,
                         force=force, verbose=verbose, normalize=normalize, composition_weight=composition_weight)


def calc_autopara_wrappable(atoms, descs, key, local=False, normalize=True, composition_weight=True, force=False, verbose=False):
    """Calculate descriptor for each config or atom

    Parameters
    ----------
    atoms: ase.atoms.Atoms / list(Atoms)
        input configuration(s)
    descs: str / list(str) / dict(Z : str / Descriptor )
        descriptor or list of descriptors (applied to all species) or dict of descriptor string for each
        species (key None for all species)
        If ``global``, combined descriptor will be concatenated.  If ``local`` and ``Z`` is not None, multiple arrays
        entries will be created, one per Zcenter, named <key>_Z_<Zcenter>.
    key: str
        key in Atoms.info (global) or Atoms.arrays (local) to store information
    local: bool, default False
        calculate a local (per-atom) descriptor, as opposed to global (per-config)
    normalize: bool, default True
        normalize final vector (e.g. if contributions from multiple species for a global were concatenated)
    composition_weight: bool, default True
        when concatenating contributions from different species for a global, weight each by composition fraction
    force: bool, default False
        overwrite key if already exists
    verbose: bool, default False
        verbose output

    Returns
    -------
    atoms: Atoms or list(Atoms)
        Input configuration(s) with descriptors in info/arrays
    """

    descs = from_any_to_Descriptor(descs, verbose=verbose)

    if isinstance(atoms, Atoms):
        at_list = [atoms]
    else:
        at_list = atoms

    # local and global share very little code - should be two functions?
    if local:
        for at in at_list:
            # check for existing key with optional _Z_<Z> suffix
            matches = [re.search(f'^{key}(_Z_[0-9]+)?$', k) for k in at.arrays]
            if any(matches):
                if force:
                    for m in matches:
                        del at.arrays[m.group(0)]
                else:
                    raise RuntimeError(f'Got arrays key {key} which already exists (perhaps '
                                       'with a _Z_<Z> suffix), pass force=True to overwrite')

            # fill in array(s)
            # this works so far in (limited) testing, but there should probably be more thought about
            # how to handle situation when various species are missing from a particular config
            # (center Z, neighbor Z, etc)
            for Zcenter in sorted(descs):
                if verbose:
                    print('doing per-atom Zcenter', Zcenter, 'actual Zs', set(at.numbers))

                combined_descs = np.zeros((len(at), 0))
                for desc in descs[Zcenter]:
                    desc_vec = desc.calc(at)['data']
                    if Zcenter is None:
                        # applies to all species
                        if desc_vec.shape[0] != len(at):
                            raise RuntimeError(('Requested local descriptor that applies to all species but '
                                                'data.shape[0] {} != len(at) == {}').format(desc_vec.shape[0], len(at)))
                        use_desc_vec = desc_vec
                    else:
                        # applies to one species
                        if desc_vec.shape[1] == 0:
                            # species isn't present
                            continue
                        elif desc_vec.shape[0] == np.sum(at.numbers == Zcenter):
                            # species is present and number of entries matches
                            use_desc_vec = np.zeros((len(at), desc_vec.shape[1]))
                            use_desc_vec[at.numbers == Zcenter, :] = desc_vec
                        else:
                            raise RuntimeError(
                                'Got descriptor for Zcenter={} with dimension {}'.format(Zcenter, desc_vec.shape[0]) +
                                ' not equal to number of that species present {}'.format(np.sum(at.numbers == Zcenter)))
                    combined_descs = np.concatenate((combined_descs, use_desc_vec), axis=1)

                if combined_descs.shape[1] > 0:
                    if Zcenter is None:
                        use_key = key
                    else:
                        use_key = f'{key}_Z_{Zcenter}'
                    at.new_array(use_key, combined_descs)
    else:
        for at in at_list:
            if key in at.info:
                if force:
                    del at.info[key]
                else:
                    raise RuntimeError(f'Got info key {key} which already exists, pass force=True to overwrite')

            # create combined vector
            combined_vec = []
            for Zcenter in sorted(descs.keys()):
                if verbose:
                    print('doing per-config Zcenter', Zcenter, 'actual Zs', set(at.numbers))
                if Zcenter is not None and composition_weight:
                    Zcenter_weight = np.sum(at.numbers == Zcenter) / len(at)
                else:
                    Zcenter_weight = 1.0
                for desc in descs[Zcenter]:
                    desc_vec = desc.calc(at)['data']
                    if desc_vec.shape[0] != 1:
                        raise RuntimeError(
                            'Requested global descriptor but data.shape[0] == {} != 1'.format(desc_vec.shape[0]))
                    combined_vec.extend(desc_vec[0, :] * Zcenter_weight)

            if normalize:
                combined_vec /= np.linalg.norm(combined_vec)
            at.info[key] = combined_vec

    return atoms
