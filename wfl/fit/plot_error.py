import re
import numpy as np
import matplotlib.pyplot as plt

units_dict = {
    "energy/atom": {"parity": "eV/at", "error": "meV/at"},
    "energy": {"parity": "eV", "error": "meV"},
    "forces": {"parity": "eV/Å", "error": "meV/Å"},
    "virial": {"parity": "eV", "error": "meV"},
    "virial/atom": {"parity": "eV/at", "error": "meV/at"}
}
def units(prop, plt_type):
    if "virial" in prop:
        prop = re.sub(r"/comp\b", "", prop)
    if "forces" in prop:
        prop = "forces"
    try:
        return units_dict[prop][plt_type]
    except KeyError:
        return "unknown units"


def annotate_parity(ax, property, ref_property_prefix, calc_property_prefix, show_legend, error_type):
    if show_legend:
        ax.legend(title=f"{error_type} per category")
    ax.set_title(f"{property} ")
    ax.set_xlabel(f"{ref_property_prefix}{property}, {units(property, 'parity')}")
    ax.set_ylabel(f"{calc_property_prefix}{property}, {units(property, 'parity')}")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lims = (min([xmin, ymin])-0.1, max([xmax, ymax])+0.1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, c='k', linewidth=0.8)

    ax.grid(color='lightgrey', ls=':')


def annotate_error(ax, property, ref_property_prefix, calc_property_prefix):
    ax.set_title(f"{property} error")
    ax.set_xlabel(f"{ref_property_prefix}{property}, {units(property, 'parity')}")
    ax.set_ylabel(f"{calc_property_prefix}{property} error, {units(property, 'error')}") 
    ax.grid(color='lightgrey', ls=':')
    ax.set_yscale('log')


def scatter(all_errors, all_diffs, all_parity, output='parity.png', ref_property_prefix=None, calc_property_prefix=None, error_type="RMSE"):

    assert error_type in ["RMSE", "MAE"], f"'error_type' must be 'RMSE' or 'MAE', not {error_type}."

    if ref_property_prefix is None:
        ref_property_prefix = "reference "
    if calc_property_prefix is None:
        calc_property_prefix = "calculated "

    num_rows = 2   # one for parity, one for errors
    num_columns = len(all_errors.keys()) # two plots for each property
    side = 4.5
    fig = plt.figure(figsize=(side * num_columns, side * num_rows))
    gs = fig.add_gridspec(num_rows, num_columns, wspace=0.25, hspace=0.25)


    props = list(all_errors.keys())
    num_cat = len(list(all_errors[props[0]].keys()))
    
    if num_cat < 11:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(idx) for idx in np.linspace(0, 1, 10)]
        show_legend = True 
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(idx) for idx in np.linspace(0, 1, num_cat)]
        show_legend = False

    for prop_idx, ((prop, differences), ref_vals, pred_vals, errors) \
        in enumerate(zip(all_diffs.items(), all_parity["ref"].values(), all_parity["calc"].values(), all_errors.values())):

        ax_parity = fig.add_subplot(gs[0, prop_idx])
        ax_error = fig.add_subplot(gs[1, prop_idx], sharex=ax_parity)

        for cat_idx, category in enumerate(differences.keys()):
            if category == "_ALL_":
                continue
            
            color = colors[cat_idx]
                
            label = f'{category}: {errors[category][error_type]*1e3:.2f} {units(prop, "error")}'
            ax_parity.scatter(ref_vals[category], pred_vals[category], label=label, edgecolors=color, facecolors='none')
            ax_error.scatter(ref_vals[category], np.array(differences[category])*1e3, edgecolors=color, facecolors='none')

        annotate_parity(ax_parity, prop, ref_property_prefix, calc_property_prefix, show_legend, error_type)
        annotate_error(ax_error, prop, ref_property_prefix, calc_property_prefix)

    plt.savefig(output, dpi=300)








