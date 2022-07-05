"""
Utility to plot 2-body binding curves for GAP models, with baseline models and differences as well
"""

import ase.io
import numpy as np
from ase.data import chemical_symbols
from matplotlib import gridspec, pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from wfl.utils.misc import chunks
from wfl.utils.parallel import construct_calculator_picklesafe

default_colors = dict(gap='tab:red', glue='tab:blue', diff='tab:cyan')


def plot_2b_multipage(gap_calculator, baseline_calculator, fig_filename, e0, atomic_numbers,
                      cutoff=None, verbose=False, baseline_label="glue", ylim=None, colors=None):
    """Plot 2-body binding curves

    Parameters
    ----------
    gap_calculator: : Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    baseline_calculator: : Calculator / (initializer, args, kwargs) / None
        ASE calculator or routine to call to create calculator
        None turns this off
    fig_filename : path_like
        figure filename, should be pdf, will be multipage
    e0 : dict / None
        e0 values with chemical symbols as keys
        None is the same as all zero
    atomic_numbers : list(int)
        atomic numbers to include all pairs of
    cutoff : float
    verbose : bool
    baseline_label : str, default "glue"
        label to use for baseline model,
    ylim : (low, high), default (-10, 10)
    colors : dict
        dict of colors to use on the plot

    Returns
    -------

    """
    if cutoff is None:
        cutoff = 6.0
    if ylim is None:
        ylim = (-10, 10)

    atomic_numbers = sorted(atomic_numbers)
    formula_list = []
    for i, z0 in enumerate(atomic_numbers):
        for z1 in atomic_numbers[i:]:
            formula_list.append("{}:{}".format(chemical_symbols[z0], chemical_symbols[z1]))

    x_values = np.linspace(0.1, cutoff, 50)

    gap_calculator = construct_calculator_picklesafe(gap_calculator)
    gap_energies = calc_2b_pot(x_values, gap_calculator, formula_list, e0)

    baseline_energies = None
    if baseline_calculator is not None:
        baseline_calculator = construct_calculator_picklesafe(baseline_calculator)
        baseline_energies = calc_2b_pot(x_values, baseline_calculator, formula_list)

    # plotting in chunks
    with PdfPages(fig_filename) as pdf:
        for formula_chunk in chunks(formula_list, 6):
            if verbose:
                print("plotting", formula_chunk)
            page_fig = plot_6_one_page(x_values, gap_energies, baseline_energies, formula_chunk,
                                       baseline_label=baseline_label, ylim=ylim, colors=colors)
            pdf.savefig(page_fig)
            plt.close(page_fig)


def calc_2b_pot(x_values, calculator, formula_list, e0=None):
    def _e0(formula_):
        if e0 is None:
            return 0.
        else:
            return np.sum([e0[s] for s in formula_.split(":")])

    ener = dict()
    for formula in formula_list:
        ener[formula] = []
        for dis in x_values:
            at = ase.Atoms(formula.replace(":", ""), positions=[[0., 0., 0.], [0., 0., dis]])
            at.calc = calculator

            ener[formula].append(at.get_potential_energy() - _e0(formula))

    for formula in ener.keys():
        ener[formula] = np.array(ener[formula])

    return ener


def plot_6_one_page(x_values, gap_ener, baseline_ener, formula_list, colors=None, baseline_label="glue",
                    ylim=(-10, 10)):
    if colors is None:
        colors = dict(default_colors)
    else:
        # overwrite default
        colors = dict(default_colors, **colors)

    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(15)
    gs_main = gridspec.GridSpec(3, 2, figure=fig)

    for i, formula in enumerate(formula_list):
        # TODO: make this two functions: curve and hist
        ax_pot = fig.add_subplot(gs_main[i])

        # plot both and their difference
        ax_pot.plot(x_values, gap_ener[formula], label='GAP', c=colors['gap'])
        if baseline_ener is not None:
            ax_pot.plot(x_values, baseline_ener[formula], label=baseline_label, c=colors[baseline_label],
                        linestyle='-.')
            ax_pot.plot(x_values, gap_ener[formula] - baseline_ener[formula], label=f'GAP-{baseline_label} difference',
                        c=colors['diff'], linestyle='-.')

        # add the minimum of both curves
        def plot_min(ax, x_vals, ener, c):
            arg = np.argmin(ener)
            ax.scatter(x_vals[arg], ener[arg], c=c, marker='+')
            ax.text(x_vals[arg], ener[arg], str(np.around(ener[arg], 2)))

        # apply them
        plot_min(ax_pot, x_values, gap_ener[formula], colors['gap'])
        if baseline_ener is not None:
            plot_min(ax_pot, x_values, baseline_ener[formula], colors['glue'])

        ax_pot.axhline(0, c='k', linestyle='-.', alpha=0.7)
        ax_pot.set_ylim(*ylim)
        ax_pot.set_xlim(0, np.max(x_values))
        ax_pot.set_title(formula, fontsize=20)
        ax_pot.grid(which='major', color='#CCCCCC', linestyle='--')
        ax_pot.grid(which='minor', color='#CCCCCC', linestyle=':')

        ax_pot.set_xlabel('separation / Å', fontsize=16)
        ax_pot.set_ylabel('energy / eV', fontsize=16)

        ax_pot.legend(loc=0, fontsize=16)

    fig.tight_layout()
    return fig
