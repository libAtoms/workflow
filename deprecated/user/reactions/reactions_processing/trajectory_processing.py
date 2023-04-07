from os import path

import ase.data
import ase.io
import numpy as np
from ase import neighborlist
try:
    from quippy.descriptors import Descriptor
except ModuleNotFoundError:
    pass

from scipy import sparse

from wfl import configset
from wfl.generate.optimize import run_autopara_wrappable
from user.generate.neb import neb_generic, neb_with_ts_and_irc
from user.generate.ts import calc_ts


def xyz_to_seed(filename):
    """Extract seed from xyz filename

    Seed is the dirname combined with the first part of the basename separated by "."
    Also means that seed cannot contain "."

    Parameters
    ----------
    filename: path_like

    Returns
    -------
    seed

    """

    return path.join(path.dirname(filename), path.basename(filename).split(".")[0])


def trajectory_min(configset, outputspec, calculator, minimise_kwargs=None):
    """Minimises every structure in a configset and returns the trajectories

    Parameters
    ----------
    configset: configset.Configset_in
    outputspec: configset.Configset_out
    calculator: ase.calculator.Calculator
    minimise_kwargs

    Returns
    -------

    """
    # Minimise and take the minima
    if minimise_kwargs is None:
        minimise_kwargs = {}

    all_minim_trajectories = run_autopara_wrappable(configset, calculator=calculator, **minimise_kwargs)

    outputspec.pre_write()
    for at_list in all_minim_trajectories:
        outputspec.write(at_list)
    outputspec.end_write()

    return configset.ConfigSet(input_configs=all_minim_trajectories)


def trajectory_ts(configset, outputspec, calculator, ts_kwargs=None):
    """Find TS every structure in a configset and returns the trajectories

    Parameters
    ----------
    configset: configset.Configset_in
    outputspec: configset.Configset_out
    calculator: ase.calculator.Calculator
    ts_kwargs

    Returns
    -------

    """
    # Minimise and take the minima
    if ts_kwargs is None:
        ts_kwargs = {}

    all_minim_trajectories = calc_ts(configset, calculator=calculator, **ts_kwargs)

    outputspec.pre_write()
    for at_list in all_minim_trajectories:
        outputspec.write(at_list)
    outputspec.end_write()

    return configset.ConfigSet(input_configs=all_minim_trajectories)


def trajectory_neb_ts_irc(configset, calculator, neb_kwargs=None, ts_kwargs=None, irc_kwargs=None, index_key=None):
    """If neighboring frames are different, NEB+TS+IRC is calculated on them

    Parameters
    ----------
    configset: configset.Configset_in
        minimisation trajectories in its groups, in order
        eg. output of trajectory_min
    calculator: ase.calculator.Calculator
    neb_kwargs: dict
        kwargs for neb calculators
    ts_kwargs: dict
    irc_kwargs: dict
    index_key: str
        dict key for indexing the NEB results

    Returns
    -------
    images: list(list(ase.Atoms))
    ts: list(ase.Atoms)
    irc: list(list(ase.Atoms))

    """
    if neb_kwargs is None:
        neb_kwargs = {}
    if ts_kwargs is None:
        ts_kwargs = {}
    if irc_kwargs is None:
        irc_kwargs = {}
    if index_key is None:
        index_key = "neb_index"

    minima = [atl[-1] for atl in configset.group_iter()]

    if len(minima) < 2:
        print("Not enough minimisation trajectories.")
        return

    collected_images = []
    collected_ts = []
    collected_irc = []

    for i in range(len(minima) - 1):
        start = minima[i]
        end = minima[i + 1]

        if compare_minima(start, end):
            images, ts, irc = neb_with_ts_and_irc(start, end, calculator=calculator,
                                                  neb_kwargs=neb_kwargs, ts_kwargs=ts_kwargs, irc_kwargs=irc_kwargs)

            for j, at in enumerate(images):
                at.info[index_key] = i + j / (len(images) + 1)

            collected_images.append(images)
            collected_ts.append(ts)
            collected_irc.append(irc)

    return collected_images, collected_ts, collected_irc


def trajectory_neb(configset, outputspec, calculator, neb_kwargs=None, index_key=None):
    """Minimises every Nth structure in a trajectory and returns the trajectories

    Parameters
    ----------
    configset: configset.Configset_in
        minimisation trajectories in its groups, in order
        eg. output of trajectory_min
    outputspec: configset.Configset_out
    calculator: ase.calculator.Calculator
    neb_kwargs: dict
        kwargs for neb calculators
    index_key: str
        dict key for indexing the NEB results

    Returns
    -------

    """
    if neb_kwargs is None:
        neb_kwargs = {}
    if index_key is None:
        index_key = "neb_index"

    minima = [atl[-1] for atl in configset.group_iter()]

    if len(minima) < 2:
        print("Not enough minimisation trajectories.")
        return

    outputspec.pre_write()
    for i in range(len(minima) - 1):
        start = minima[i]
        end = minima[i + 1]

        if compare_minima(start, end):
            images = neb_generic(start, end, calculator=calculator, **neb_kwargs)

            for j, at in enumerate(images):
                at.info[index_key] = i + j / (len(images) + 1)

            outputspec.write(images)

    outputspec.end_write()


def compare_minima(mol1, mol2, cutoff=6.0, threshold=0.95, zeta=2.0):
    """Compare molecules, return if they are the same

    Parameters
    ----------
    mol1: ase.Atoms
    mol2: ase.Atoms
    cutoff: float
    threshold: float
        lower bound for molecules to be accepted as the same
    zeta: float
        power for kernel

    Returns
    -------
    is_same: bool

    """

    # descriptor string -- only elements seen in this one
    all_z = np.unique([mol1.get_atomic_numbers(), mol2.get_atomic_numbers()])
    num_z = len(all_z)
    z_str = " ".join([str(z) for z in all_z])
    desc_str = f"soap cutoff={cutoff} n_max=8 l_max=6 atom_sigma=0.3 n_species={num_z} species_Z={{{z_str}}}"
    desc = Descriptor(desc_str)

    # combined metric as well
    vec1 = desc.calc_descriptor(mol1)
    vec2 = desc.calc_descriptor(mol2)
    product = np.exp(np.mean(np.log(np.sum(vec1 * vec2, axis=1) ** zeta)))

    return product < threshold


def check_number_of_connected_parts(mol, cutoff=6.0):
    """ Calculates the number of connected fragments in a configuration

    In general, we want to avoid configs where two fragments are outside of the longest cutoff, because then we can
    take them apart and add them separately. So you are looking for this to return 1 for everything in the trainingset.

    Parameters
    ----------
    mol: ase.Atoms
    cutoff: float, default=6.0

    Returns
    -------
    n_components: int
        number of components found with neighbour-list

    """
    # cutOff = neighborlist.natural_cutoffs(mol)
    # neighbor_list = neighborlist.NeighborList([cutoff / 2] * len(mol), self_interaction=False, bothways=True)
    neighbor_list = neighborlist.NeighborList(np.ones(shape=len(mol)) * cutoff / 2.,
                                              self_interaction=False, bothways=True)
    neighbor_list.update(mol)
    matrix = neighbor_list.get_connectivity_matrix()
    n_components, component_list = sparse.csgraph.connected_components(matrix)

    return n_components


def calc_structure_weights(frames, energy_std_limit=0.01):
    """Weights per structure, hardcoded for now.

    formula
    - energy_std (in eV/atom) if < 10: std ** 2 / 10, else: std

    Parameters
    ----------
    frames
    energy_std_limit: float, default=0.010
        energy limit where

    Returns
    -------

    """
    w = []

    for i, at in enumerate(frames):
        # in meV/atom them
        w.append(np.std([at.info[k] for k in at.info.keys() if "gap_committee_" in k and "_energy" in k]) / len(at))
    w = np.array(w)
    print("energy_std_limit", energy_std_limit)
    w[w < energy_std_limit] = w[w < energy_std_limit] ** 2 / energy_std_limit

    return w


def cut_trajectory_with_global_metric(frames, threshold, zeta=2.0):
    """
    Cuts the trajectory at the point where the combined similarity metric goes below threshold for the first time.
    """

    soap_3 = Descriptor(
        "soap n_max=8 l_max=6 cutoff=3.0 cutoff_transition_width=1.0 atom_sigma=0.3 n_species=3 species_Z={6 1 8 }")

    arr = np.array(soap_3.calc_descriptor(frames))
    sim_first = np.sum(arr[0, :, :] * arr, axis=2) ** zeta

    products = np.exp(np.mean(np.log(sim_first), axis=1))

    try:
        cut_index = np.argwhere(products < threshold).min()
    except ValueError:
        cut_index = -1

    return cut_index


def create_desc_ref(reference_frames, cutoff_list, z_list, desc_str_base=None):
    """Creates the reference descriptor set and calculators for use

    Parameters
    ----------
    reference_frames: list(Atoms)
        reference list of atoms objects to calculate the descriptors of
    cutoff_list: list(float) or list(str)
        cutoff values to use in SOAP kernels
    z_list: list(int)
        atomic numbers to calculate kernels on
    desc_str_base: f-str, optional
        formatable string for descriptor, formatting keys: `cutoff` and `z`
        default="soap cutoff={cutoff} n_max=8 l_max=6 atom_sigma=0.3 n_species=3 species_Z={{1 6 8}} Z={z}"

    Returns
    -------
    soap_dict:
        descriptor calculator objects in dictionary, like:
        dict(cutoff -> dict(z -> descriptor_calculator))
    desc_ref:
        descriptor arrays in a dictionary, like:
        dict(cutoff -> dict(z -> descriptor_array))

    """

    if desc_str_base is None:
        desc_str_base = "soap cutoff={cutoff} n_max=8 l_max=6 atom_sigma=0.3 n_species=3 species_Z={{1 6 8}} Z={z}"

    soap_dict = dict()
    desc_ref = dict()

    for cutoff in cutoff_list:
        assert isinstance(cutoff, str)
        soap_dict[cutoff] = dict()
        desc_ref[cutoff] = dict()
        for z in z_list:
            assert isinstance(z, int)
            desc = Descriptor(desc_str_base.format(cutoff=cutoff, z=z))

            soap_dict[cutoff][z] = desc
            desc_ref[cutoff][z] = np.concatenate([d for d in desc.calc_descriptor(reference_frames) if np.ndim(d) > 1])

    return soap_dict, desc_ref


def calc_max_similarity(desc_ref, desc_interest, zeta=2):
    """Calculate the maximal kernel similarity between reference and vectors of interest

    Parameters
    ----------
    desc_ref: array_like
        descriptor array of reference
    desc_interest: array_like
        descriptor array for interest,
        can be 2D or 3D, last dimension needs to be the descriptor's indexing to contract on
    zeta: float, default=2.0
        power to raise the kernel to

    Returns
    -------
    k_max: array_like
        maximal kernel similarity for each vector of interest with the ref.
        matches descriptor dimension as: 2D -> 1D, 3D -> 2D

    """
    k_3d = np.dot(desc_interest, desc_ref.T) ** zeta

    # minima for each frame
    if k_3d.ndim == 3:
        k_max = np.max(k_3d, axis=(1, 2))
    else:
        k_max = np.max(k_3d)

    return k_max


def calc_max_similarity_atoms(at, soap_dict, desc_ref):
    all_species = np.unique(at.get_atomic_numbers())

    for z in all_species:
        for cutoff in soap_dict.keys():
            desc_trajectory = soap_dict[cutoff][z].calc_descriptor(at)
            max_similarity = calc_max_similarity(desc_ref[cutoff][z], desc_trajectory)
            at.info['max_similarity_{}_{}'.format(ase.data.chemical_symbols[z], cutoff)] = max_similarity

    return at


def extract_data_from_trajectories(filenames):
    """Process a trajectory into dict-arrays for plotting and selection operations

    Parameters
    ----------
    filenames

    Returns
    -------

    """
    # todo: add indexing, or use ConfigSet
    data = dict()
    for fn in filenames:
        # read frames
        frames = ase.io.read(fn, ":")
        len_traj = len(frames)

        # symbols to work with
        symbols = np.array(frames[0].get_chemical_symbols())
        only_species = np.unique(symbols)
        similarity_keys = [k for k in frames[0].info.keys() if "max_similarity" in k]

        shape = (len(frames))
        data[fn] = dict(evar=np.zeros(shape=shape),
                        x=np.arange(len_traj),
                        formula=frames[0].get_chemical_formula()
                        )

        for sym in only_species:
            data[fn]["fvar_max_{}".format(sym)] = np.zeros(shape=shape)
            data[fn]["fvar_mean_{}".format(sym)] = np.zeros(shape=shape)

        for k in similarity_keys:
            data[fn][k] = np.zeros(shape=shape)

        for i, at in enumerate(frames):
            data[fn]["evar"][i] = np.std([at.info[k] for k in at.info.keys() if "energy_" in k]) / len(at)

            force_arr = np.array([at.arrays[k] for k in at.arrays.keys() if "force_" in k or k == "forces"])
            var_arr = np.var(force_arr, axis=0)

            for sym in only_species:
                data[fn]["fvar_max_{}".format(sym)][i] = np.sqrt(np.max(var_arr[symbols == sym]))
                data[fn]["fvar_mean_{}".format(sym)][i] = np.sqrt(np.mean(var_arr[symbols == sym]))
            for k in similarity_keys:
                data[fn][k][i] = at.info[k]
    return data
