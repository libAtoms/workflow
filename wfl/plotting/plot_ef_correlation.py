"""
Utilities for plotting the performance of a GAP model
"""

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

try:
    from wfl.plotting import maxveit_plottools as plottools
except ModuleNotFoundError:
    plottools = None
    print("Plottools cannot be imported, some functionality will be missing")

chemical_accuracy = 0.04336  # eV / atom
default_colors = dict(energy='tab:blue', force_H='tab:green', force_C='tab:gray', force_O='tab:red',
                      chemical_accuracy='lightsteelblue')
default_thin_params = dict(force_thin_r=0.4, force_weight_power=0.4,
                           pt_base_size=6.0)  # pt_base_size: this is the matplotlib default


def extract_energies_per_atom(frames, energy_parameter_name, e0):
    """Get the energies from a list of atoms, per atom

    Parameters
    ----------
    frames : list(ase.Atoms) / ase.Atoms / Configset_in
    energy_parameter_name : dict_key
        key in atoms.info for energy
    e0 : None / dict
        shift by e0, keys need to be chemical symbols
        None triggers no shift at all, still divides by number of atoms

    Returns
    -------
    energy_array : array_like(float)
    """

    if isinstance(frames, ase.Atoms):
        frames = [frames]

    if e0 is None:
        # using zero so not summing over them at all
        energy_array = np.array([at.info[energy_parameter_name] / len(at) for at in frames])
    elif isinstance(e0, dict):
        # sum e0 and divide by number of atoms after that
        energy_array = np.array(
            [(at.info[energy_parameter_name] - np.sum([e0[s] for s in at.get_chemical_symbols()])) / len(at) for at in
             frames])
    else:
        raise ValueError('given e0 was not understood')

    return energy_array


def extract_forces(frames, force_parameter_name, flat=True):
    """Extracts components of forces and gives you a dictionary of them.

    Parameters
    ----------
    frames : list(ase.Atoms) / ase.Atoms / Configset_in
    force_parameter_name : dict_key
        key in atoms.info for energy
    flat : bool
        flat (n_atoms * 3) or (n_atoms, 3) array of the forces

    Returns
    -------
    force_components : dict
        dict of np arrays containing force components

    """
    if isinstance(frames, ase.Atoms):
        frames = [frames]

    results = dict()

    for at in frames:
        if len(at) == 1:
            # skipping the single atoms
            continue
        force_components = at.arrays[force_parameter_name]
        symbols = at.get_chemical_symbols()

        for j, sym in enumerate(symbols):
            if sym not in results.keys():
                # add the key if needed
                results[sym] = []
            results[sym].append(force_components[j])

    # make flat numpy arrays out of them
    for key, val in results.items():
        if flat:
            results[key] = np.array(val).flat[:]
        else:
            results[key] = np.array(val)

    return results


def rms_text_for_plots(x_ref, x_pred, unit='meV/Å', digits=2, latex=False):
    """Compile RMSE text for plots

    Parameters
    ----------
    x_ref : array_like
    x_pred : array_like
    unit : str, {"eV/Å", "eV/A", "meV/Å", "meV/A", "meV", "eV"}
    digits
    latex

    Returns
    -------

    """

    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)

    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError('WARNING: not matching shapes in rms')

    error_2 = (x_ref - x_pred) ** 2
    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))

    if latex:
        pm = "\pm"
    else:
        pm = "±"

    unit.strip()
    if unit in ['meV/Å', 'meV/A', 'meV']:
        average *= 1000
        std_ *= 1000
    elif unit in ['eV/Å', 'eV/A', 'eV']:
        pass
    else:
        raise ValueError(f"Unit not understood: {unit}")

    text = "RMSE \n {average: .{digits}f} {pm} {std: .{digits}f} {unit}".format(average=average, digits=digits,
                                                                                unit=unit, pm=pm, std=std_)

    return text


def plot_energy(fig, gs_slot, ref_energy, gap_energy, show_chemical_accuracy=True, colors=None, color_points=True):
    # this plots the energy with Max Veit's thinpoints

    if colors is None:
        colors = default_colors
    else:
        colors = default_colors.update(colors)

    rms_text = rms_text_for_plots(ref_energy, gap_energy, unit="meV")
    abs_difference = np.abs(gap_energy - ref_energy)

    # setting xlim_both, 10% added on both ends
    low = np.min((ref_energy.min(), gap_energy.min()))
    high = np.max((ref_energy.max(), gap_energy.max()))
    diff = np.abs(high - low)
    xlim_both = np.array((low - 0.1 * diff, high + 0.1 * diff))

    # add 10% on both ends in log scale
    err_low = abs_difference.min()
    err_high = abs_difference.max()
    err_diff = np.log10(err_high) - np.log10(err_low)
    ylim_err = np.power(10, (np.log10(err_low) - 0.1 * err_diff, np.log10(err_high) + 0.1 * err_diff))

    # now making a pair of plots: upper for the potential and the lower for the hist
    # todo: make sure the top ones (corr) are squares
    gs_local = gs_slot.subgridspec(nrows=2, ncols=1, hspace=0., height_ratios=(2, 1))
    ax_corr = fig.add_subplot(gs_local[0, 0])
    ax_err = fig.add_subplot(gs_local[1, 0])

    if color_points:
        # colormap for the
        cm = plt.cm.get_cmap('tab20b')
        c_arr = np.arange(len(ref_energy))
        sc1 = ax_corr.scatter(ref_energy, gap_energy, label='energy', marker='.', c=c_arr, cmap=cm)
        ax_err.scatter(ref_energy, abs_difference, label='energy', marker='.', c=c_arr, cmap=cm)
        # adding the colorbar
        fig.colorbar(sc1, ax=ax_err, orientation="horizontal")
    else:
        ax_corr.scatter(ref_energy, gap_energy, label='energy', marker='.', color=colors['energy'])
        ax_err.scatter(ref_energy, abs_difference, label='energy', marker='.', color=colors['energy'])

    ax_corr.plot(xlim_both, xlim_both, c='k')

    if show_chemical_accuracy:
        # highlighting the chemical accuracy
        ax_corr.fill_between(xlim_both, xlim_both - chemical_accuracy, xlim_both + chemical_accuracy,
                             color=colors['chemical_accuracy'], alpha=0.5)
        ax_err.fill_between(xlim_both, (chemical_accuracy, chemical_accuracy), color=colors['chemical_accuracy'],
                            alpha=0.5)

    # adding text
    ax_corr.text(0.1, 0.9, rms_text, transform=ax_corr.transAxes, fontsize='large', verticalalignment='top')

    # fancy settings
    ax_corr.set_xlim(xlim_both)
    ax_corr.set_ylim(xlim_both)
    ax_corr.set_ylabel('GAP energy / (eV/atom)')
    ax_corr.grid(which='major', color='#CCCCCC', linestyle='--')
    ax_corr.grid(which='minor', color='#CCCCCC', linestyle=':')
    ax_corr.set_axisbelow(True)
    ax_corr.set_xticklabels([])

    ax_err.set_xlabel('Reference energy / (eV/atom)')
    ax_err.set_ylabel('|GAP energy error| / (eV/atom)')
    ax_err.set_ylim(ylim_err)
    ax_err.set_yscale('log')
    ax_err.set_xlim(xlim_both)
    ax_err.grid(which='major', color='#CCCCCC', linestyle='--')
    ax_err.grid(which='minor', color='#CCCCCC', linestyle=':')
    ax_err.set_axisbelow(True)


def plot_force_components(fig, gs_slot, force_ref_array, force_gap_array, label, ticks='right',
                          force_weight_power=0.4, force_thin_r=0.4, pt_base_size=6.0, colors=None, **kwargs):
    if colors is None:
        colors = default_colors

    # this plots the forces and errors with thinpoints
    rms_text = rms_text_for_plots(force_ref_array, force_gap_array, unit="meV/Å", digits=1)
    abs_difference = np.abs(force_ref_array - force_gap_array)

    thin_corr, weights_corr = _thin(force_ref_array, force_gap_array, r=force_thin_r)
    thin_err, weights_err = _thin(force_ref_array, abs_difference, do_y_log=True, r=force_thin_r)

    # setting xlim for both
    xlim_both = np.array((-1.0, 1.0)) * 20.

    # add 10% on both ends in log scale
    err_low = thin_err[:, 1].min()
    err_high = thin_err[:, 1].max()
    err_diff = np.log10(err_high) - np.log10(err_low)
    ylim_err = np.power(10, (np.log10(err_low) - 0.1 * err_diff, np.log10(err_high) + 0.1 * err_diff))

    # now making a pair of plots: upper for the potential and the lower for the hist
    gs_local = gs_slot.subgridspec(nrows=2, ncols=1, hspace=0., height_ratios=(2, 1))
    ax_corr = fig.add_subplot(gs_local[0, 0])
    ax_err = fig.add_subplot(gs_local[1, 0])

    # Plot thin points
    try:
        c = colors['force_{}'.format(label)]
    except KeyError:
        # penalty then
        c = 'k'
    plottools.scatter_thin_points(thin_corr, weights_corr ** force_weight_power, ax=ax_corr, s=pt_base_size, alpha=0.5,
                                  edgecolor='k', label=label, c=c)
    ax_corr.plot(xlim_both, xlim_both, c='k')
    plottools.scatter_thin_points(thin_err, weights_err ** force_weight_power, ax=ax_err, s=pt_base_size, alpha=0.5,
                                  edgecolor='k', label=label, c=c)

    # fancy settings
    ax_corr.set_ylabel('GAP force / (eV/Å)', labelpad=2.)

    ax_corr.set_xlim(xlim_both)
    ax_corr.set_ylim(xlim_both)
    ax_corr.legend(loc=0, fontsize="large")

    ax_err.set_yscale('log')
    ax_err.set_ylim(ylim_err)
    ax_err.set_xlim(xlim_both)
    ax_err.set_xlabel('DFT force / (eV/Å)')
    ax_err.set_ylabel('|GAP force error| / (eV/Å)')

    ax_corr.grid(which='major', color='#CCCCCC', linestyle='--')
    ax_corr.grid(which='minor', color='#CCCCCC', linestyle=':')
    ax_err.grid(which='major', color='#CCCCCC', linestyle='--')
    ax_err.grid(which='minor', color='#CCCCCC', linestyle=':')
    ax_corr.set_axisbelow(True)
    ax_err.set_axisbelow(True)
    ax_corr.set_xticklabels([])

    if ticks == "right":
        ax_corr.yaxis.tick_right()
        ax_corr.yaxis.set_label_position('right')
        ax_err.yaxis.tick_right()
        ax_err.yaxis.set_label_position('right')
    elif ticks == "left":
        ax_corr.yaxis.tick_left()
        ax_corr.yaxis.set_label_position('left')
        ax_err.yaxis.tick_left()
        ax_err.yaxis.set_label_position('left')

    ax_corr.text(0.9, 0.1, rms_text, transform=ax_corr.transAxes, fontsize='large', horizontalalignment='right',
                 verticalalignment='bottom')


def _thin(ref, pred, do_x_log=False, do_y_log=False, **kwargs):
    """
    Thins the forces
    """
    data = np.column_stack((np.array(ref).flat, np.array(pred).flat))

    if do_x_log and do_y_log:
        # all rows with any zeros in them
        data = data[~np.any(data == 0., axis=1)]
    elif do_x_log:
        # only if the first column has zeros
        data = data[~(data[:, 0] == 0.), :]
    elif do_y_log:
        # only if the second column has zeros
        data = data[~(data[:, 1] == 0.), :]

    try:
        thin, weights = plottools.thin_transformed(data, do_x_log=do_x_log, do_y_log=do_y_log, **kwargs)
    except ValueError:
        print('Value error with thinpoints. Using uniform weighting. Parameters: do_x_log={}, do_y_log={}, {}'.format(
            do_x_log, do_y_log, kwargs))
        thin = data
        weights = np.ones_like(thin)
    return thin, weights


def plot(frames, fig_filename, e0, ref_energy_key="DFT_energy", gap_energy_key="gap_energy",
         ref_force_key="DFT_forces", gap_force_key="gap_forces", colors=None, thin_params=None):
    if colors is None:
        colors = default_colors
    else:
        colors = default_colors.update(colors)

    if thin_params is None:
        thin_params = default_thin_params
    else:
        thin_params = default_thin_params.update(thin_params)

    energy_ref = extract_energies_per_atom(frames, ref_energy_key, e0)
    energy_gap = extract_energies_per_atom(frames, gap_energy_key, e0)

    # remove single atoms
    single_atom_text = None
    deleted = np.zeros((energy_ref.shape[0],), dtype=bool)
    for i, at in enumerate(frames):
        if len(at) == 1:
            if single_atom_text is None:
                single_atom_text = 'Difference on single atoms:'
            deleted[i] = True
            single_atom_text += '\n {}: {:.3} meV'.format(at.get_chemical_symbols()[0], energy_gap[i] * 1000)
    # now remove these for the rest of the work
    energy_ref = energy_ref[~deleted]
    energy_gap = energy_gap[~deleted]

    # take forces
    forces_ref = extract_forces(frames, ref_force_key)
    forces_gap = extract_forces(frames, gap_force_key)

    # plotting --------------------------------------------
    with PdfPages(fig_filename) as pdf:
        # energy plot
        fig_ener = plt.figure(figsize=(10, 13))
        gs_ener = gridspec.GridSpec(1, 1, figure=fig_ener)
        plot_energy(fig_ener, gs_ener[0, 0], energy_ref, energy_gap, show_chemical_accuracy=True)

        if isinstance(single_atom_text, str):
            fig_ener.axes[0].text(0.1, 0.8, single_atom_text, transform=fig_ener.axes[0].transAxes, fontsize='large',
                                  verticalalignment='top')
        fig_ener.tight_layout()
        pdf.savefig(fig_ener)
        plt.close(fig_ener)

        # force plots
        for label in forces_ref.keys():
            fig_force = plt.figure(figsize=(10, 13))
            gs_force = gridspec.GridSpec(1, 1, figure=fig_force)
            plot_force_components(fig_force, gs_force[0, 0], forces_ref[label], forces_gap[label],
                                  label=label, ticks="right", colors=colors, **thin_params)

            fig_force.tight_layout()
            pdf.savefig(fig_force)
            plt.close(fig_force)
