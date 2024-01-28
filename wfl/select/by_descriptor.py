import warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator, svds


def do_svd(at_descs, num, do_vectors='vh'):
    def mv(v):
        return np.dot(at_descs, v)

    def rmv(v):
        return np.dot(at_descs.T, v)

    # noinspection PyArgumentList
    A = LinearOperator(at_descs.shape, matvec=mv, rmatvec=rmv, matmat=mv)

    return svds(A, k=num, return_singular_vectors=do_vectors)


def _hashable_struct_data(at):
    return (at.numbers.tobytes(), at.positions.tobytes(), at.pbc.tobytes(), at.cell.tobytes(),
            at.get_initial_magnetic_moments().tobytes())


def CUR(mat, num, stochastic=True, rng=None, exclude_list=None):
    """Compute selection by CUR of descriptors with dot-product, with optional exponentiation

    Parameters
    ----------
    mat: np.array(vec_len, n_vecs) or (n_vecs, n_vecs)
        rectangular array of descriptors as column vectors or square kernel matrix
    num: int
        number to select
    stochastic: bool, default True
        use stochastic selection algorithm
    rng: numpy.random.Generator
        random number generator, required if stochastic
    exclude_list: list(int), default None
        list of descriptor indices to exclude

    Returns
    -------
    selected_inds: list(int)
        list of indices for selected descriptors
    """

    (u, s, vt) = do_svd(mat, min(max(1, int(num / 2)), min(mat.shape) - 1))

    c_scores = np.sum(vt ** 2, axis=0) / vt.shape[0]

    if exclude_list is not None:
        c_scores[exclude_list] = 0.0
        c_scores /= np.sum(c_scores)

    if sum(c_scores > 0) < num:
        raise RuntimeError(f'Asked for {num} configs but only {sum(c_scores > 0)} are available')

    if stochastic:
        selected_inds = rng.choice(range(mat.shape[1]), size=num, replace=False, p=c_scores)
    else:
        selected_inds = np.argsort(c_scores)[-num:]

    return selected_inds, c_scores[selected_inds]


def prep_descs_and_exclude(inputs, at_descs, at_descs_info_key, exclude_list):
    """process configs and/or input descriptor row array and exclude list to produce
    descriptors column array and indices of excluded configuration

    Parameters
    ----------
    inputs: ConfigSet
        input configurations
    at_descs: np.ndarray( n_configs x desc_len ), default None
        if not None, array of descriptors (as rows) for each config
        mutually exclusive with at_descs_info_key, one is required
    at_descs_info_key: str, default None
        key into Atoms.info dict for descriptor vector of each config
        mutually exclusive with at_descs, one is required
    exclude_list: iterable(Atoms)
        list of Atoms structures to be excluded from selection, to be converted into
        indices into enumerate(inputs)

    Returns
    -------
        at_descs_cols: np.ndarray (desc_len x n_configs)
            array of descriptors (as columns) for each config
        exclude_ind_list: list(int)
            indices of configurations to exclude
    """
    assert sum([at_descs is None, at_descs_info_key is None]) == 1

    if exclude_list is not None:
        # create list of hashable data as a set to fast comparison
        exclude_list = {_hashable_struct_data(at) for at in exclude_list}

    # make array of column vectors of descriptors
    exclude_ind_list = []
    if at_descs is None or exclude_list is not None:
        # go through inputs once, extract descriptors and/or create exclude list indices
        if at_descs is None:
            at_descs = []
        for at_i, at in enumerate(inputs):
            if at_descs_info_key:
                # need to extract into at_descs
                at_descs.append(at.info[at_descs_info_key])
            if exclude_list is not None and _hashable_struct_data(at) in exclude_list:
                # Indexing is simplified by including all atoms in at_descs, and letting
                # actual selection (CUR, FPS, etc) exclude ones that should be excluded.
                exclude_ind_list.append(at_i)

    # make at_descs be column vectors (desc_len x n_descs)
    at_descs = np.asarray(at_descs).T

    return at_descs, exclude_ind_list


def write_selected_and_clean(inputs, outputs, selected, at_descs_info_key=None, keep_descriptor_info=True):
    """Writes selected (by index) configs to output configset

    Parameters
    ----------
    inputs: ConfigSet
        input configuration set
    outputs: OutputSpec
        target for output of selected configurations
    selected: list(int)
        list of indices to be selected, cannot have duplicates
    at_descs_info_key: str, default None
        key in info dict to delete if keep_descriptor_info is False
    keep_descriptor_info: bool, default True
        keep descriptor in info dict
    """
    if not keep_descriptor_info and at_descs_info_key is None:
        raise RuntimeError('Got False \'keep_descriptor_info\' but not the info key \'at_descs_info_key\' to wipe')

    selected_s = set(selected)
    assert len(selected) == len(selected_s)

    counter = 0
    # inputs is an iterable, can't directly reference specific # configs,
    # so loop through and search in set (should be fast)
    for at_i, at in enumerate(inputs):
        if at_i in selected_s:
            if not keep_descriptor_info:
                del at.info[at_descs_info_key]
            outputs.store(at)
            counter += 1
            if counter >= len(selected_s):
                # skip remaining iterator if we've used entire selected list
                break
    outputs.close()


def CUR_conf_global(inputs, outputs, num, at_descs=None, at_descs_info_key=None, kernel_exp=None, stochastic=True,
                    rng=None, keep_descriptor_info=True, exclude_list=None, center=True,
                    leverage_score_key=None):
    """Select atoms from a list or iterable using CUR on global (per-config) descriptors

    Parameters
    ----------
    inputs: ConfigSet
        atomic configs to select from
    outputs: OutputSpec
        where to write output to
    num: int
        number to select
    rng: int, default None
        random number generator
    at_descs: np.array(n_descs, desc_len), mutually exclusive with at_descs_info_key
        list of descriptor vectors
    at_descs_info_key: str, mutually exclusive with at_descs
        key to Atoms.info dict containing per-config descriptor vector
    kernel_exp: float, default None
        exponent to compute kernel (if other than 1)
    stochastic: bool, default True
        use stochastic selection
    keep_descriptor_info: bool, default True
        do not delete descriptor from info
    exclude_list: iterable(Atoms)
        list of Atoms to exclude from CUR selection.  Needs to be _exactly_ the same as
        actual Atoms objects in inputs, to full machine precision
    center: bool, default True
        center data before doing SVD, as generally required for PCA
    leverage_score_key: str, default None
        if not None, info key to store leverage score in

    Returns
    -------
    ConfigSet corresponding to selected configs output
    """
    if outputs.all_written():
        warnings.warn('output is done, returning')
        return outputs.to_ConfigSet()

    at_descs, exclude_ind_list = prep_descs_and_exclude(inputs, at_descs, at_descs_info_key, exclude_list)

    # do SVD on kernel if desired
    if kernel_exp is not None:
        descs_mat = np.matmul(at_descs.T, at_descs) ** kernel_exp
        if center:
            # centering like that used for kernel-PCA
            row_of_col_means = np.mean(descs_mat, axis=0)
            col_of_row_means = np.mean(descs_mat, axis=1)
            descs_mat -= row_of_col_means
            descs_mat = (descs_mat.T - col_of_row_means).T
            descs_mat += np.mean(col_of_row_means)
    else:
        if center:
            descs_mat = (at_descs.T - np.mean(at_descs, axis=1)).T
        else:
            descs_mat = at_descs

    selected, _ = CUR(mat=descs_mat, num=num, stochastic=stochastic,
                      rng=rng, exclude_list=exclude_ind_list)

    write_selected_and_clean(inputs, outputs, selected, at_descs_info_key, keep_descriptor_info)

    return outputs.to_ConfigSet()


def greedy_fps_conf_global(inputs, outputs, num, at_descs=None, at_descs_info_key=None,
                           keep_descriptor_info=True, exclude_list=None,
                           prev_selected_descs=None, O_N_sq=False, rng=None, verbose=False):
    """Select atoms from a list or iterable using greedy farthest point selection on global (per-config) descriptors

    Parameters
    ----------
    inputs: ConfigSet
        atomic configs to select from
    outputs: OutputSpec
        where to write output to
    num: int
        number to select
    at_descs: np.array(n_descs, desc_len), mutually exclusive with at_descs_info_key
        list of descriptor vectors
    at_descs_info_key: str, mutually exclusive with at_descs
        key to Atoms.info dict containing per-config descriptor vector
    keep_descriptor_info: bool, default True
        do not delete descriptor from info
    exclude_list: iterable(Atoms)
        list of Atoms to exclude from selection by descriptor
    prev_selected_descs: np.array(n_prev_descs, desc_len), default False
        if present, list of previously selected descriptors to also be farthest from
    O_N_sq: bool, default False
        use O(N^2) algorithm with smaller prefactor
    rng: numpy.random.Generator
        random number generator
    verbose: bool, default False
        more verbose output

    Returns
    -------
    selected_configs : ConfigSet
        corresponding to selected configs output
    """
    if outputs.all_written():
        warnings.warn(f'output {outputs} is done, returning')
        return outputs.to_ConfigSet()

    if prev_selected_descs is not None and not isinstance(prev_selected_descs, np.ndarray):
        prev_selected_descs = np.asarray(prev_selected_descs)

    at_descs, exclude_ind_list = prep_descs_and_exclude(inputs, at_descs, at_descs_info_key, exclude_list)
    # at_descs is returned as column vectors

    n_avail = at_descs.shape[1] - len(exclude_ind_list)
    if n_avail < num:
        raise RuntimeError(f'Asked for {num} configs but only {n_avail} are available')

    if O_N_sq:
        # actually calculate full (N+N_prev)xN similarities
        if prev_selected_descs is not None and len(prev_selected_descs) > 0:
            # also calculate similarities to descs of previously selected
            lhs = np.vstack([prev_selected_descs, at_descs.T])
            # the fact that this 1st len(prev_selected_descs) are the ones that were previously
            # selected is being relied on below
            prev_selected = list(range(len(prev_selected_descs)))
        else:
            lhs = at_descs.T
            prev_selected = []
        # rows are previously selected and currently available, columns are only currently available
        similarities = np.matmul(lhs, at_descs)

        max_similarity = np.max(similarities) + 1.0

        # set high value for excluded indices
        similarities[:, exclude_ind_list] = max_similarity

        if len(prev_selected) == 0:
            # nothing to compute distance to, initialize randomly from those not in exclude list
            p = np.ones(similarities.shape[1])
            p[exclude_ind_list] = 0.0
            p /= np.sum(p)
            selected_indices = [rng.choice(range(similarities.shape[1]), p=p)]
            similarities[:, selected_indices[-1]] = max_similarity
        else:
            selected_indices = []

        if verbose:
            print('initial selection', selected_indices)
        while len(selected_indices) < num:
            # For each available config (column), find closest of those already selected (max sim over axis 0).
            # Note that similarities matrix has previously selected stacked at lowest index values, so
            #   selected_indices (whch indexes into avail configs, i.e. columns) needs to be offset by that
            #   much to correctly index into rows
            similarities_to_nearest_selected = np.max(
                similarities[prev_selected + [s + len(prev_selected) for s in selected_indices]], axis=0)
            # Of available configs, find the one that is farthest (argmin) from those previously selected
            farthest_available = np.argmin(similarities_to_nearest_selected)
            if verbose:
                print('farthest nearest and similarities to nearest', farthest_available,
                      similarities_to_nearest_selected)
            selected_indices.append(farthest_available)

            # set similarities so it doesn't get selected again
            similarities[:, selected_indices[-1]] = max_similarity

    else:
        # calculate only needed dot products

        # this assumes a kernel, i.e. in [0,1] - needs to be done some other way if
        # max possible similarity is not 1.0
        # maybe it would be safer to do this as a _distance_, where the minimum is definitely 0
        max_similarity = 2.0

        if prev_selected_descs is not None and len(prev_selected_descs) > 0:
            selected_indices = []
            similarities_arr = prev_selected_descs @ at_descs
        else:
            # nothing to compute distance to, initialize randomly from those not in exclude list
            p = np.ones(at_descs.shape[1])
            p[exclude_ind_list] = 0.0
            p /= np.sum(p)
            selected_indices = [rng.choice(range(at_descs.shape[1]), p=p)]
            similarities_arr = np.asarray([at_descs[:, selected_indices[-1]].T @ at_descs])
            similarities_arr[:, selected_indices[-1]] = max_similarity

        # rows are prev selected and currently available, cols are currently available
        similarities_arr[:, exclude_ind_list] = max_similarity

        if verbose:
            print('initial selection', selected_indices)
        while len(selected_indices) < num:
            # for each available config (columns), find nearest (max along axis 0) already selected
            similarities_to_nearest_selected = np.max(similarities_arr, axis=0)

            # over each config, find one that is farthest from nearest already selected
            farthest_available = np.argmin(similarities_to_nearest_selected)
            if verbose:
                print('farthest nearest and similarities to nearest', farthest_available,
                      similarities_to_nearest_selected)
            selected_indices.append(farthest_available)

            # calculate dot product of most recently selected relative to every other available
            similarity_row = at_descs[:, selected_indices[-1]].T @ at_descs
            # set high similarity to excluded configs so they are not selected
            similarity_row[exclude_ind_list] = max_similarity

            # add to rectangular matrix of similarities of all previously selected
            similarities_arr = np.vstack([similarities_arr, similarity_row])

    write_selected_and_clean(inputs, outputs, selected_indices, at_descs_info_key, keep_descriptor_info)

    return outputs.to_ConfigSet()
