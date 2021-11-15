import sys

import numpy as np


def _select_by_bin(weights, bin_edges, quantities, n, kT, verbose=False):
    if verbose:
        print('got histogram', len(weights), weights)

    if 0 >= n:
        raise ValueError("Not defined for non-positive n")

    if kT <= 0:
        kT = np.Infinity

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers -= bin_centers[0]

    # this will fail for non-uniform bins
    total_n = 0
    n_per_bin = 0
    while total_n < n:
        n_per_bin += 1
        n_from_bin = [min(int(w), int(np.round(n_per_bin * np.exp(-bin_ctr / kT)))) for bin_ctr, w in
                      zip(bin_centers, weights)]
        total_n = sum(n_from_bin)
    # noinspection PyUnboundLocalVariable
    # this not being defined is caught somewhere up if n <= 0
    n_from_bin = np.asarray(n_from_bin)

    # always remove excess from most occupied bins
    # not actually best if using Boltzmann bias
    n_excess = total_n - n
    while n_excess > 0:
        bin_max = np.max(n_from_bin)
        max_bins = np.where(n_from_bin == bin_max)[0]
        n_from_bin[max_bins[np.random.choice(range(len(max_bins)), min(n_excess, len(max_bins)), replace=False)]] -= 1
        n_excess = sum(n_from_bin) - n

    assert sum(n_from_bin) == n
    assert all(n_from_bin >= 0)

    if verbose:
        print('got n_from_bin', n_from_bin)

    bin_of_quantities = np.searchsorted(bin_edges[1:], quantities)
    selected_inds = []
    for bin_i in range(len(weights)):
        bin_list = np.where(bin_of_quantities == bin_i)[0]
        selected_inds.extend(np.random.choice(bin_list, n_from_bin[bin_i], replace=False))

    return selected_inds


def _select_by_individual_weight(weights, bin_edges, quantities, n, kT, verbose=False):
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
    return np.random.choice(np.arange(len(config_prob)), n, replace=False, p=config_prob)


def _select_indices_flat_boltzmann_biased(quantities, n, kT=-1.0, bins='auto', by_bin=True, verbose=False):
    """Select samples by Boltzmann-weight biased flat histogram

    Parameters
    ----------
    quantities: iterable(num) w/ len()
        quantities to histogram/bias
    n: int
        number of samples to return
    kT: float, default -1
        if > 0 temperature to bias by
    bins: np.histogram argument, int or sequence of scalars or str, default 'auto'
        value passed to np.histogram bins argument
    by_bin: bool, default True
        do selection by bin, as opposed to setting probability for each config and trying to select that way
        produces better flat histograms
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
        return _select_by_bin(weights, bin_edges, quantities, n, kT, verbose)
    else:
        return _select_by_individual_weight(weights, bin_edges, quantities, n, kT, verbose)


def biased_select_conf(inputs, outputs, num, info_field, kT=-1.0, bins='auto', by_bin=True, verbose=False):
    if outputs.is_done():
        sys.stderr.write('Returning from {__name__} since output is done\n')
        return outputs.to_ConfigSet_in()

    quantities = []
    avail_inds = {}
    for at_i, at in enumerate(inputs):
        try:
            quantities.append(at.info[info_field])
            avail_inds[at_i] = len(quantities) - 1
        except KeyError:
            pass

    # convert to set for faster checking (O(1)?) of "in" below
    selected_indices = set(
        _select_indices_flat_boltzmann_biased(quantities, num, kT, bins=bins, by_bin=by_bin, verbose=verbose))

    for at_i, at in enumerate(inputs):
        try:
            if avail_inds[at_i] in selected_indices:
                outputs.write(at)
        except KeyError:
            # skip configs that are not in avail_inds
            pass

    outputs.end_write()
    return outputs.to_ConfigSet_in()
