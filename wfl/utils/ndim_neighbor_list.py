import numpy as np


def calc_list(positions, ranges, cartesian_distance=True):
    """simple and naive neighbor list in arbitrary dimensions

    Parameters
    ----------
    positions: ndarray(n_positions, n_dims)
        Array of positions to compute neighbors of
    ranges: ndarray(n_dims)
        Ranges to use as neighbor cutoff in each dimension
    cartesian_distance: bool, default True
        Calculate neighbor list using Cartesian distance (i.e. nearby hyperoval).
        If False, use a rectilinear prism, i.e. distance in every dimension must be less than range.

    Returns
    -------
        i, j: list(neighbor_pairs), list(neighbor_pairs)
            corresponding lists of i-j neighbor pairs indices
    """

    ii = []
    jj = []
    for i in range(len(positions)):
        dval = (positions[:] - positions[i]) / ranges
        if cartesian_distance:
            nearby = np.where(np.linalg.norm(dval, axis=1) < 1.0)[0]
        else:
            nearby = np.where(np.all(np.abs(dval) < 1.0, axis=1))[0]
        ii.extend([i] * len(nearby))
        jj.extend(nearby)

    return np.asarray(ii), np.asarray(jj)


def calc_list_cells(positions, ranges, Cartesian_distance=True):
    """Linear scaling (at constant _density_) neighbor list in arbitrary dimensions,
    based on core of ase.neighborlist.neighbor_list

    Parameters
    ----------
    positions: ndarray(n_positions, n_dims)
        array of positions to compute neighbors of
    ranges: ndarray(n_dims)
        ranges to use as neighbor cutoff in each dimension
    Cartesian_distance: bool, default True
        Calculate neighbor list using Cartesian distance (i.e. nearby hyperoval).
        If False, use a rectilinear prism, i.e. distance in every dimension must be
        less than range.

    Returns
    -------
        i, j: list(neighbor_pairs), list(neighbor_pairs)
            corresponding lists of i-j neighbor pairs indices
    """

    # normalize positions so that all start at 0, and cutoff is always 1
    positions = np.array(positions) / ranges
    positions -= np.min(positions, axis=0)

    # from ase.neighborlist.neighbor_list
    nbins_c = np.floor(np.max(positions, axis=0)).astype(int) + 1
    nbins = np.prod(nbins_c)

    n_dims = positions.shape[1]

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search = np.array([1] * n_dims)

    # Sort configs into bins.
    bin_index_ic = np.floor(positions).astype(int)

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = bin_index_ic[:, n_dims - 1]
    for i_dim in range(n_dims - 2, -1, -1):
        bin_index_i *= nbins_c[i_dim]
        bin_index_i += bin_index_ic[:, i_dim]

    # config_i contains config index in new sort order.
    config_i = np.argsort(bin_index_i)
    bin_index_i = bin_index_i[config_i]

    # Find max number of configs per bin
    max_nconfigs_per_bin = np.max(np.bincount(bin_index_i))

    # Sort configs into bins: configs_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of configs inside that bin. This list is
    # homogeneous, i.e. has the same size *max_nconfigs_per_bin* for all bins.
    # The list is padded with -1 values.
    configs_in_bin_ba = -np.ones([nbins, max_nconfigs_per_bin], dtype=int)
    for i in range(max_nconfigs_per_bin):
        # Create a mask array that identifies the first config of each bin.
        mask = np.append([True], bin_index_i[:-1] != bin_index_i[1:])
        # Assign all first configs.
        configs_in_bin_ba[bin_index_i[mask], i] = config_i[mask]

        # Remove configs that we just sorted into configs_in_bin_ba. The next
        # "first" config will be the second and so on.
        mask = np.logical_not(mask)
        config_i = config_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all configs have been sorted into bins.
    assert len(config_i) == 0
    assert len(bin_index_i) == 0

    # Now we construct neighbor pairs by pairing up all configs within a bin or
    # between bin and neighboring bin. config_pairs_pn is a helper buffer that
    # contains all potential pairs of configs between two bins, i.e. it is a list
    # of length max_nconfigs_per_bin**2.
    config_pairs_pn = np.indices((max_nconfigs_per_bin, max_nconfigs_per_bin),
                                 dtype=int)
    config_pairs_pn = config_pairs_pn.reshape(2, -1)

    # Initialized empty neighbor list buffers.
    first_at_neightuple_nn = []
    second_at_neightuple_nn = []

    # This is the main neighbor list search. We loop over neighboring bins and
    # then construct all possible pairs of configs between two bins, assuming
    # that each bin contains exactly max_nconfigs_per_bin configs. We then throw
    # out pairs involving pad configs with config index -1 below.
    binD_A = np.meshgrid(*[np.arange(nbins_c[i_dim])
                           for i_dim in range(n_dims - 1, -1, -1)], indexing='ij')
    binD_A = np.flip(np.asarray(binD_A), axis=0)
    # The memory layout of binD_A is such that computing
    # the respective bin index leads to a linearly increasing consecutive list.
    # The following assert statement succeeds:
    #     b_b = (binx_xyz + nbins_c[0] * (biny_xyz + nbins_c[1] *
    #                                     binz_xyz)).ravel()
    #     assert (b_b == np.arange(np.prod(nbins_c))).all()

    # First configs in pair.
    _first_at_neightuple_n = configs_in_bin_ba[:, config_pairs_pn[0]]
    for i_offset in range(np.product(2 * neigh_search + 1)):
        dD = []
        for i_dim in range(n_dims):
            dD.append(i_offset % (2 * neigh_search[i_dim] + 1))
            i_offset //= (2 * neigh_search[i_dim] + 1)
        dD -= neigh_search

        neighbinD_A = [np.mod(binD_A[i_dim] + dD[i_dim], nbins_c[i_dim])
                       for i_dim in range(n_dims)]

        neighbin_b = neighbinD_A[n_dims - 1]
        for i_dim in range(n_dims - 2, -1, -1):
            neighbin_b *= nbins_c[i_dim]
            neighbin_b += neighbinD_A[i_dim]
        neighbin_b = neighbin_b.ravel()

        # Second config in pair.
        _second_at_neightuple_n = configs_in_bin_ba[neighbin_b][:, config_pairs_pn[1]]

        # We have created too many pairs because we assumed each bin
        # has exactly max_nconfigs_per_bin configs. Remove all superfluous
        # pairs. Those are pairs that involve an config with index -1.
        mask = np.logical_and(_first_at_neightuple_n != -1,
                              _second_at_neightuple_n != -1)
        if mask.sum() > 0:
            first_at_neightuple_nn += [_first_at_neightuple_n[mask]]
            second_at_neightuple_nn += [_second_at_neightuple_n[mask]]

    # Flatten overall neighbor list.
    first_at_neightuple_n = np.concatenate(first_at_neightuple_nn)
    second_at_neightuple_n = np.concatenate(second_at_neightuple_nn)

    # Sort neighbor list.
    i = np.argsort(first_at_neightuple_n)
    first_at_neightuple_n = first_at_neightuple_n[i]
    second_at_neightuple_n = second_at_neightuple_n[i]

    # Compute distance vectors.
    distance_vector_nc = positions[second_at_neightuple_n] - positions[first_at_neightuple_n]
    abs_distance_vector_n = np.sqrt(np.sum(distance_vector_nc * distance_vector_nc, axis=1))

    # We have still created too many pairs. Only keep those with distance
    # smaller than 1.0.
    if Cartesian_distance:
        # this is original expression, oval in space
        mask = abs_distance_vector_n < 1.0
    else:
        # alternative is a rectangle in space
        mask = np.max(np.abs(distance_vector_nc), axis=1) < 1

    first_at_neightuple_n = first_at_neightuple_n[mask]
    second_at_neightuple_n = second_at_neightuple_n[mask]

    return first_at_neightuple_n, second_at_neightuple_n
