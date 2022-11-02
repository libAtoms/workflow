import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def annotate_parity(ax, property):
    ax.legend(f"RMS per category")
    ax.set_title(f"{property} arity plot")
    ax.set_xlabel(f"Reference {property}, units")
    ax.set_ylabel(f"Predicted {property}, units")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lims = (min([xmin, ymin])-0.1, max([xmax, ymax])+0.1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, c='k', linewidth=0.8)

    ax.grid(color='lightgrey', ls=':')


def annotate_error(ax, property):
    ax.set_title(f"{property} error plot")
    ax.set_xlabel(f"Reference {property}, units")
    ax.set_ylabel("error, units") 
    ax.axhline(0, color='k', lw=0.8)
    ax.grid(color='lightgrey', ls=':')


def scatter(all_errors, all_diffs, all_parity, output='scatter.png'):

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
            label = f'{category}: {errors[category]["RMS"]:.4f} add_units'
            ax_parity.scatter(ref_vals[category], pred_vals[category], label=label, color=color)
            ref = list(np.concatenate(ref_vals[category]))
            if prop=="forces" and category == "rad":
                import pdb; pdb.set_trace()
            # print(prop, category)
            ax_error.scatter(ref, differences[category], color=color)

        annotate_parity(ax_parity, all_errors)
        annotate_error(ax_error, all_errors)

    plt.savefig(output, dpi=300)








