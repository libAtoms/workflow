import numpy as np
import matplotlib.pyplot as plt

units_dict = {
    "energy/atom": {"parity": "eV/at", "error": "meV/at"},
    "energy": {"parity": "eV", "error": "meV"},
    "forces": {"parity": "eV/Å", "error": "meV/Å"}
}
def units(prop, plt_type):
    if "forces" in prop:
        prop = "forces"
    try:
        return units_dict[prop][plt_type]
    except KeyError:
        return "unknown units"


def annotate_parity(ax, property, ref_property_prefix, calc_property_prefix):
    ax.legend(title=f"RMSE per category")
    ax.set_title(f"{property} parity plot")
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
    ax.set_title(f"{property} error plot")
    ax.set_xlabel(f"{ref_property_prefix}{property}, {units(property, 'parity')}")
    ax.set_ylabel(f"{calc_property_prefix}{property} error, {units(property, 'error')}") 
    ax.grid(color='lightgrey', ls=':')
    ax.set_yscale('log')


def scatter(all_errors, all_diffs, all_parity, output='scatter.png', ref_property_prefix=None, calc_property_prefix=None):

    if ref_property_prefix is None:
        ref_property_prefix = "reference "
    if calc_property_prefix is None:
        calc_property_prefix = "calculated "

    num_rows = 2   # one for parity, one for errors
    num_columns = len(all_errors.keys()) # two plots for each property
    side = 4.5
    fig = plt.figure(figsize=(side * num_columns, side * num_rows))
    gs = fig.add_gridspec(num_rows, num_columns)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.linspace(0, 1, 10)]

    for prop_idx, ((prop, differences), ref_vals, pred_vals, errors) \
        in enumerate(zip(all_diffs.items(), all_parity["ref"].values(), all_parity["calc"].values(), all_errors.values())):

        ax_parity = fig.add_subplot(gs[0, prop_idx])
        ax_error = fig.add_subplot(gs[1, prop_idx], sharex=ax_parity)

        for cat_idx, category in enumerate(differences.keys()):
            if category == "_ALL_":
                continue
            
            color = colors[cat_idx]
                
            label = f'{category}: {errors[category]["RMS"]*1e3:.2f} {units(prop, "error")}'
            ax_parity.scatter(ref_vals[category], pred_vals[category], label=label, color=color)
            ax_error.scatter(ref_vals[category], np.array(differences[category])*1e3, color=color)

        annotate_parity(ax_parity, prop, ref_property_prefix, calc_property_prefix)
        annotate_error(ax_error, prop, ref_property_prefix, calc_property_prefix)

    plt.savefig(output, dpi=300)








