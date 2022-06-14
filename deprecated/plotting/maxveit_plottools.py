# vim: set encoding=utf-8
"""
This file was taken from https://github.com/max-veit/pyutils/blob/master/plottools.py
written my Max Veit for plotting

Plot tools: Reusable utilities for working with matplotlib plots.

Contents:
    - params_(reports|poster|...) Standard parameter sets for final-quality plots
    - colors              Easily distinguishable color sets

Functions:
    - plot_smoothed       Plot a time series smoothed with a cosine kernel
    - plot_acorr          Plot the autocorrelation of a timeseries
    - thin_points         Thin data points that are too close to each other
    - thin_transformed    Thin data points in a transformed space
    - scatter_thin_points Plot a set of thinned data points with weight info
    - scatter_mark_outliers Mark outliers at the edges of a scatterplot
    - scatter_outliers_size Make a scatterplot with sizes and outliers

"""

from __future__ import unicode_literals, print_function, division

import logging

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot
from scipy import signal
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

# Some colours for easily distinguishable and colourblind-friendly coding
# Use the 4th colour if necessary; may not be colourblind-friendly though
# from colorbrewer.org
cb_colors = {}
cb_colors['green'] = '#1b9e77'
cb_colors['orange'] = '#d95f02'
cb_colors['purple'] = '#7570b3'
cb_colors['pink'] = '#e7298a'
cb_color_cycle = mpl.rcsetup.cycler(
    'c', [cb_colors[key] for key in ['purple', 'orange', 'green', 'pink']])


def plot_smoothed(time, data, avg_period, ax=None, **plot_args):
    """Plot a time series smoothed with a cosine kernel

    Parameters:
        time        Array of equally spaced time points
        data        Array of data to plot at those times
        avg_period  Number of time points to average over
                    (width of the averaging kernel)
        ax          Axes to plot into (default: current/new Axes)
        plot_args   Any additional arguments to pass to pyplot.plot()
    """
    logger.info("Averaging period: {:g} fs".format(
        avg_period * (time[1] - time[0])))
    # krnl = np.ones(avg_period) / avg_period
    krnl = signal.cosine(avg_period)
    krnl = krnl / np.sum(krnl)
    data_ma = np.convolve(data, krnl, mode='valid')
    if ax is None:
        ax = pyplot.gca()
    ax.plot(time[avg_period / 2:-(avg_period / 2 - 1)], data_ma, **plot_args)


def plot_acorr(time, data, step_start, sample_dt, corr_guess, axes=None,
               **plot_args):
    """Plot the autocorrelation of a timeseries and show its integral

    The integral is useful for estimating the error on the mean of a
    correlated timeseries, see Sokal's lecture on Monte Carlo methods for
    more information.

    Parameters
    ----------
        time:        Time values of the timeseries
                    Must be sequential and equally spaced
        data:        Data of the timeseries
        step_start:  Step (array index) at which to start computing
                    averages and correlation
        sample_dt:   Time between two successive samples
        corr_guess:  Guess for the correlation time of the series
        axes:        Matplotlib axes to plot into.  Leaving it as None will
                    plot into the current axes (or create a new one if none
                    exists yet)
        **plot_args: keyword arguments are passed to the plot() command used to
                     create the autocorrelation trace.

    Returns
    -------
        mean:        Mean of data in range (after `step_start`)
        serr:        Standard error on mean
                    (accounting for autocorrelation)
        n_eff:       Number of effective independent samples
        corr_time:   Integrated correlation time
    """
    logger.info("Starting from time {:.2f}".format(time[step_start]))
    data_mean = np.mean(data[step_start:])
    data_zeromean = data[step_start:] - np.mean(data[step_start:])
    n = len(data_zeromean)
    dataacorr = (signal.correlate(data_zeromean, data_zeromean, 'same')
                 / (n - np.abs(np.arange(n) - n // 2)))
    window_width = int(corr_guess * 6.0 / sample_dt)
    acorr_int = 0.5 * np.sum(
        dataacorr[n // 2 - window_width: n // 2 + window_width + 1]) * sample_dt
    corr_time = acorr_int / dataacorr[n // 2]
    serr_mean = np.sqrt(2.0 * acorr_int / n / sample_dt)
    n_eff = n / 2 / corr_time * sample_dt
    if axes is None:
        axes = pyplot.gca()
    axes.plot(time[step_start:] - time[n // 2 + step_start],
              dataacorr)
    axes.grid()
    axes.axvspan(-1. * time[window_width], time[window_width], fc='k', alpha=0.2,
                 label="Integration region")
    axes.axvspan(-1. * corr_time, corr_time, fc='r', alpha=0.3,
                 label="Correlation time")
    axes.set_xlim(-12.0 * corr_guess, 12.0 * corr_guess)
    return data_mean, serr_mean, n_eff, corr_time


def thin_points(data, r=None, nmax=1, density=None, len_scale=0.01):
    """Thin a set of data points down to some target density.

    Uses "poisonous seedling" thinning algorithm: Pick a point at
    random, and if it has more than a certain number of neighbours
    within some predefined radius, randomly remove neighbours until the
    number is reached.  Repeat in turn for all the remaining points.

    If the data has an extreme aspect ratio (either the ratio or its
    reciprocal greater than about 1.5), or is to be plotted with one or
    both axes on a log scale, the function thin_transformed() will usually
    produce better results (NB only for data in 2-D space).

    Parameters
    ----------
    data:        Data matrix, size (N,d) - N is num points, d is problem
                 dimensionality
    density:     Target density, maximum density in thinned set, mutually
                 exclusive with r
    len_scale:   Length scale of density averaging, as a fraction of the
                 range in the data (maximum range along any dimension).
                 Default 0.01, optional for density.
    r:           Absolute length scale of distance averaging (radius of
                 sphere within which neighbours are counted)
    nmax:        Maximum number of neighbours for a point (default 1), optional
                 for r

    Note that the default of nmax=1 ensures that no two points are
    closer than r, thinning points to a maximum (approximately uniform)
    density of 1/r^dim.  This mode of use seems to produce the best
    results in practice.

    NB density only implemented in two dimensions (i.e. d==2).

    """
    dim = data.shape[1]
    if density is None and r is None:
        raise ValueError("Must specify r if not using density")
    elif density is not None and dim != 2:
        raise NotImplementedError("Density not implemented for dimension != 2")
    elif density is not None:
        # Choose nmax and r such that r is about 1/100 of the range of
        # the data, but make nmax an integer.
        data_range = np.max(np.max(data, axis=0) - np.min(data, axis=0))
        nmax = np.ceil(density * np.pi * (data_range * len_scale) ** 2)
        r = np.sqrt(nmax / (density * np.pi))
    # Technically the leafsize should be equal to the ratio of volumes of the
    # d-cube to the d-sphere times nmax, but I think this is good enough.
    neightree = KDTree(data, leafsize=max(20, int(nmax) * 2))
    deleted = np.zeros((data.shape[0],), dtype=bool)
    point_weight = np.ones((data.shape[0],), dtype=np.float32)
    idces = np.arange(data.shape[0])
    np.random.shuffle(idces)
    for data_idx in idces:
        if deleted[data_idx]:
            continue
        # TODO adjust eps for efficiency?
        neighbours = neightree.query_ball_point(data[data_idx, :], r, eps=0.0)
        neighbours.remove(data_idx)
        if nmax == 1:
            del_idces = neighbours
        else:
            del_idces = np.random.choice(neighbours,
                                         len(neighbours) - nmax + 1)
        deleted[del_idces] = True
        point_weight[data_idx] += np.sum(point_weight[del_idces])
        point_weight[del_idces] = 0
    return data[~deleted, :], point_weight[~deleted]


def thin_transformed(data_full, equalize_aspect=True, aspect=None,
                     do_y_log=False, do_x_log=False, **thin_params):
    """Apply the thinning algorithm in a transformed 2-D data space

    Data points are always returned in the original (untransformed)
    space

    Parameters
    ----------
        equalize_aspect:   Equalize the aspect ratio (decided by the
                          range of the data, or aspect if given)
                          before thinning
        aspect:            Manually specify aspect ratio (width / height)
                          of the data (or log-data if thinning in log space)
        do_y_log:          Take the log of the data (y-coordinates)
                          before thinning
        do_x_log:          Take the log of the data (x-coordinates)
                          before thinning
        **thin_params: keyword arguments passed on to thin_points().  Note in
                       particular that the 'r' parameter is in X axis units.
    """
    if data_full.shape[1] != 2:
        raise ValueError("This function is designed to work in 2-D space; "
                         "got data.shape[1] == {:d} instead.".format(
            data_full.shape[1]))
    data_trans = data_full.copy()
    if do_y_log:
        data_trans[:, 1] = np.log(np.abs(data_trans[:, 1]))  # tks32 add abs
    if do_x_log:
        data_trans[:, 0] = np.log(np.abs(data_trans[:, 0]))  # tks32 add abs
    if equalize_aspect and (aspect is None):
        data_range = np.max(data_trans, axis=0) - np.min(data_trans, axis=0)
        aspect = data_range[0] / data_range[1]
    if equalize_aspect:
        data_trans[:, 1] = data_trans[:, 1] * aspect
    data_thin, weights = thin_points(data_trans, **thin_params)
    # Undo the transformations
    if equalize_aspect:
        data_thin[:, 1] = data_thin[:, 1] / aspect
    if do_x_log:
        data_thin[:, 0] = np.exp(data_thin[:, 0])
    if do_y_log:
        data_thin[:, 1] = np.exp(data_thin[:, 1])
    return data_thin, weights


def scatter_thin_points(data_thin, weights, method='size', ax=None,
                        **plot_args):
    """Make a scatterplot with a thinned set of points

    Parameters
    ---------
    data_thin:   Thinned data, e.g. from thin_points()
    weights:     Weights of the thinned data, e.g. from thin_points()
                (represents how many real points each thin point
                represents
    method:      How to represent the weights; options are 'size',
                'color', and 'alpha'
    ax:          Existing axis to plot into, if any
    **plot_args: keyword arguments passed to pyplot.scatter().
    """
    if ax is None:
        ax = pyplot.gca()
    if method == 'size':
        # TODO An automatic determination of the base size based on the
        #      actual plot size (in pixels) might be nice...
        s_scale = plot_args.pop('s', 2)
        # Retain the border (linewidth) to make it easier to see points' sizes
        pl = ax.scatter(data_thin[:, 0], data_thin[:, 1],
                        s=(weights * s_scale), **plot_args)
    elif method == 'color':
        plot_args.pop('c', None)
        pl = ax.scatter(data_thin[:, 0], data_thin[:, 1], linewidth=0, c=weights,
                        **plot_args)
    elif method == 'alpha':
        alpha_min = 0.0
        c_base = plot_args.pop('c', 'k')
        c_base = mpl.colors.colorConverter.to_rgba(c_base)  # tks32
        c_array = np.zeros((len(weights), 4))
        c_array[:, :] = c_base
        c_array[:, 3] = 1.0 + (alpha_min - 1.0) * np.exp(-1.0 * weights)
        pl = ax.scatter(data_thin[:, 0], data_thin[:, 1], linewidth=0, c=c_array,
                        **plot_args)
    else:
        raise ValueError("Unknown representation method " + method)
    return pl


def scatter_mark_outliers(xs, ys, xlim, ylim, ax=None, **plot_args):
    """Mark outliers at the edges of a scatterplot

    The outliers are marked at the edge of the graph as large red dots,
    by default size 40, color 'r'.  If a label was present, the outliers
    will be labelled as the same with " (outliers)" appended.

    Parameters
    ---------
    xs:      List of x-coordinates of points, passed to pyplot.scatter()
    ys:      List of y-coordinates of points, passed to pyplot.scatter()
    xlim:    Limits of x-axis, as a tuple (xmin, xmax)
    ylim:    Limits of y-axis, as a tuple (ymin, ymax)
    ax:      Existing axis to plot into, if any
    **plot_args: keyword arguments passed to pyplot.scatter().

    Returns
    -------
    plts: collection of scatterplots - (top, bottom, left, right).
    """

    if ax is None:
        ax = pyplot.gca()
    outl_top = xs[(ys > ylim[1]) & (xs >= xlim[0]) & (xs <= xlim[1])]
    outl_bot = xs[(ys < ylim[0]) & (xs >= xlim[0]) & (xs <= xlim[1])]
    # Let out_left and outl_right handle the (literal) corner cases
    outl_left = np.array(ys[(xs < xlim[0])])
    outl_right = np.array(ys[(xs > xlim[1])])
    outl_left[outl_left < ylim[0]] = ylim[0]
    outl_left[outl_left > ylim[1]] = ylim[1]
    outl_right[outl_right < ylim[0]] = ylim[0]
    outl_right[outl_right > ylim[1]] = ylim[1]
    if 'linewidth' not in plot_args:
        plot_args['linewidth'] = 0
    if 'c' not in plot_args:
        plot_args['c'] = 'r'
    if 's' not in plot_args:
        plot_args['s'] = 40
    plot_args['clip_on'] = False
    plot_args['label'] = (plot_args['label'] + ' (outliers)'
                          if 'label' in plot_args else '_')
    op_t = ax.scatter(outl_top, ylim[1] * np.ones_like(outl_top), **plot_args)
    plot_args['label'] = '_'
    op_b = ax.scatter(outl_bot, ylim[0] * np.ones_like(outl_bot), **plot_args)
    op_l = ax.scatter(
        xlim[0] * np.ones_like(outl_left), outl_left, **plot_args)
    op_r = ax.scatter(
        xlim[1] * np.ones_like(outl_right), outl_right, **plot_args)
    # Not typically necessary, but just ensure the outlier markers end
    # up at the edge of the plot
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return op_t, op_b, op_l, op_r


def scatter_outliers_size(data_thin, weights, xlim, ylim, ax=None, c_out='r',
                          s_out=40, **plot_args):
    """Do scatterplot with outliers, encoding the given weights as size.

    NB This doesn't change the size of the outlier markers - it might be
    possible by modifying the original escatter_outliers() function, but
    does it even make sense to encode the weights of outliers?

    Parameters
    ----------
    data_thin:   Thinned data, e.g. from thin_points()
    weights:     Weights of the thinned data, e.g. from thin_points()
                 (represents how many real points each thin point
                 represents
    xlim:        Limits of x-axis, as a tuple (xmin, xmax)
    ylim:        Limits of y-axis, as a tuple (ymin, ymax)
    ax:          Existing axis to plot into, if any
    c_out:       Colour to use to mark outliers, default 'r'
    s_out:       Size for outlier markers, default 40 (px)
    **plot_args: keyword arguments that will only modify the properties of
                 the central (non-outlier) scatterplot.

    Returns
    -------
    plts: the original plot as well as the four outlier scatterplots -
          (orig, top, bottom, left, right)

    """
    pl = scatter_thin_points(data_thin, weights, ax=ax, **plot_args)
    outps = scatter_mark_outliers(
        *data_thin.T, xlim=xlim, ylim=ylim, ax=ax, c=c_out, s=s_out)
    return (pl,) + outps
