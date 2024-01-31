import sys

import numpy as np


def _select_by_bin(weights, bin_edges, quantities, n, rng, kT=-1.0, replace=False, verbose=False):
    if verbose:
        print('got histogram', len(weights), weights)

    assert not replace

    if n <= 0:
        raise ValueError("Not defined for non-positive n")

    if kT is None or kT <= 0:
        kT = np.Infinity

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers -= bin_centers[0]

    # This will fail for non-uniform bins
    total_n = 0
    prefactor_l = 0
    prefactor_h = 1
    # find prefactor_l and prefactor_h that bracket minimal prefactor to Boltzmann weights
    # sufficient to get enough configs
    while total_n < n:
        n_from_bin = [min(int(w), int(np.round(prefactor_h * np.exp(-bin_ctr / kT)))) for bin_ctr, w in
                      zip(bin_centers, weights)]
        total_n = sum(n_from_bin)
        if total_n < n:
            prefactor_l = prefactor_h
            prefactor_h *= 2
    # binary search for optimal prefactor
    while prefactor_h - prefactor_l > 1.0e-10 * prefactor_h:
        prefactor_m = (prefactor_l + prefactor_h) / 2.0
        n_from_bin = [min(int(w), int(np.round(prefactor_m * np.exp(-bin_ctr / kT)))) for bin_ctr, w in
                      zip(bin_centers, weights)]
        total_n = sum(n_from_bin)
        if total_n < n:
            prefactor_l = prefactor_m
        else:
            prefactor_h = prefactor_m

    # get final numbers from prefactor_h, which is large enough to get enough configs
    n_from_bin = [min(int(w), int(np.round(prefactor_h * np.exp(-bin_ctr / kT)))) for bin_ctr, w in
                  zip(bin_centers, weights)]
    n_from_bin = np.asarray(n_from_bin)

    # always remove excess from most occupied bins
    # not actually best if using Boltzmann bias
    n_excess = total_n - n
    while n_excess > 0:
        bin_max = np.max(n_from_bin)
        max_bins = np.where(n_from_bin == bin_max)[0]
        n_from_bin[max_bins[rng.choice(range(len(max_bins)), min(n_excess, len(max_bins)), replace=False)]] -= 1
        n_excess = sum(n_from_bin) - n

    assert sum(n_from_bin) == n
    assert all(n_from_bin >= 0)

    if verbose:
        print('got n_from_bin', n_from_bin)

    bin_of_quantities = np.searchsorted(bin_edges[1:], quantities)
    selected_inds = []
    for bin_i in range(len(weights)):
        bin_list = np.where(bin_of_quantities == bin_i)[0]
        selected_inds.extend(rng.choice(bin_list, n_from_bin[bin_i], replace=False))

    return selected_inds


def _select_by_individual_weight(weights, bin_edges, quantities, n, rng, kT=-1.0, replace=False, verbose=False):
    min_quantity = np.min(quantities)

    # if this out of range then histogram is broken
    bin_i = np.searchsorted(bin_edges[1:], quantities)
    config_prob = 1.0 / weights[bin_i]
    config_prob[np.isnan(config_prob)] = 0.  # convert any NaN to 0.0 to skip empty bins

    if kT > 0.0:
        config_prob *= np.exp(-(np.array(quantities) - min_quantity) / kT)

    config_prob /= np.sum(config_prob)

    # Doesn't work well when n is a significant fraction of len(prob),
    # even if enough samples are available to respect probabilities,
    # but manual implementation isn't any better.
    return rng.choice(np.arange(len(config_prob)), n, replace=replace, p=config_prob)


def _select_indices_flat_boltzmann_biased(quantities, n, rng, kT=-1.0, bins='auto', by_bin=True, replace=False, verbose=False):
    """Select samples by Boltzmann-weight biased flat histogram

    Parameters
    ----------
    quantities: iterable(num) w/ len()
        quantities to histogram/bias
    n: int
        number of samples to return
    rng: np.random.Generator
        random number generator
    kT: float, default -1
        if > 0 temperature to bias by
        [kT] should have the same unit as the "quantities" parameter
    bins: np.histogram argument, int or sequence of scalars or str, default 'auto'
        value passed to np.histogram bins argument
    by_bin: bool, default True
        do selection by bin, as opposed to setting probability for each config and trying to select that way
        produces better flat histograms
    replace: bool, default False
        do selection with replacement
    verbose: bool, default True
        verbose output

    Returns
    -------
    list of selected indices into quantities list
    """

    if n > len(quantities):
        raise RuntimeError(f'More configs requested {n} than available {len(quantities)}')

    weights, bin_edges = np.histogram(quantities, bins=bins)

    if by_bin:
        return _select_by_bin(weights, bin_edges, quantities, n, rng=rng, kT=kT, replace=replace, verbose=verbose)
    else:
        return _select_by_individual_weight(weights, bin_edges, quantities, n, rng=rng, kT=kT, replace=replace, verbose=verbose)


def biased_select_conf(inputs, outputs, num, info_field, rng, kT=-1.0, bins='auto', by_bin=True, replace=False, verbose=False):
    """select configurations by Boltzmann biased flat histogram on some quantity in Atoms.info

    Parameters
    ----------
    inputs: ConfigSet
        input configurations
    output: OutputSpec
        output configurations
    num: int
        number of configs to select
    info_field: string
        Atoms.info key for quantity by which to do flat histogram and Boltzmann bias
    rng: np.random.Generator
        random number generator
    kT: float, default -1
        Boltzmann bias temperature, <= 0 to not bias
        [kT] should have the same unit as the "info_field" parameter
    bins: np.histogram bins argument, default 'auto'
        argument to pass to np.histogram
    by_bin: bool, default True
        do selections by bin, which is more accurate, but works badly for small kT and does not allow for selection with replacement
    replace: bool, default False
        do selection with replacement (i.e. repeat configs)
    verbose: bool, default False
        verbose output

    Returns
    -------
    ConfigSet containing output configs
    """

    if outputs.all_written():
        sys.stderr.write('Returning from {__name__} since output is done\n')
        return outputs.to_ConfigSet()

    quantities = []
    avail_inds = {}
    for at_i, at in enumerate(inputs):
        try:
            quantities.append(at.info[info_field])
            avail_inds[at_i] = len(quantities) - 1
        except KeyError:
            pass

    # convert to set for faster checking (O(1)?) of "in" below
    selected_indices = _select_indices_flat_boltzmann_biased(quantities, num, rng=rng, kT=kT, bins=bins,
                                                             by_bin=by_bin, replace=replace, verbose=verbose)

    selected_indices = sorted(selected_indices)
    selected_i = 0
    for at_i, at in enumerate(inputs):
        while selected_i < len(selected_indices) and selected_indices[selected_i] <= at_i:
            if selected_indices[selected_i] == at_i:
                outputs.store(at)
            selected_i += 1
        if selected_i >= len(selected_indices):
            break

    outputs.close()
    return outputs.to_ConfigSet()
