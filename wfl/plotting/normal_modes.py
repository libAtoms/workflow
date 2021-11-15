import os
import copy

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from ase.units import invcm

from wfl.generate_configs import vib

def cmap_in_alpha(color, number_of_slices=256):
    """Constructs a matplotlib colormap from a single color,
       varying transparency"""
    rgb = mpl.colors.to_rgba(color)
    my_cmap = np.repeat([rgb], number_of_slices, axis=0)
    my_cmap[:, -1] = np.linspace(0, 1, number_of_slices)
    my_cmap = mpl.colors.ListedColormap(my_cmap)
    return my_cmap


def eigenvector_plot(nm1, nm2, prop_prefix1, prop_prefix2, label1=None,
                     label2=None, fig_fname='eigenvector_plot.png',
                     arrange_by='frequency', cmap='darkred',
                     adjust_color_range=False, tolerance=1e-4):
    """Eigenvector x eigenvector plot

    Parameters
    ----------
    nm1: str / Atoms
        Atoms with normal mode information or file to read that from
    nm2: str / Atoms
        Atoms with normal mode information or file to read that from
    prop_prefix1: str
        Prefix for normal modes-related properties in atoms.info/arrays
    prop_prefix2: str
        Prefix for normal modes-related properties in atoms.info/arrays
    label1: str, default None
        Plot label for nm1
    label2: str, default None
        Plot label for nm2
    fig_fname: str, default eigenvector_plot.png
        figure name to save as
    arrange_by: 'order'/'frequency', default 'frequency'
        whether to plot a heatmap of eigenvectors in order or a scatterplot
        of frequency vs frequency
    cmap: str, default='darkred'
        matplotlib colormap, colormap name or color name. If color name is
        given, colormap in transparency is constructed
    adjust_color_range: bool, default False
        whether to normalise colormap range to best describe data or fix to
        [0, 1] region
    tolerance: float, default 1e-4
        allowance to for upper and lower bound outside of [0, 1] for dot
        products


    """

    assert arrange_by in ['order', 'frequency']

    if label1 is None:
        if isinstance(nm1, str):
            label1 = os.path.splitext(os.path.basename(nm1))[0]
        else:
            label1 = 'method 1'
    if label2 is None:
        if isinstance(nm2, str):
            label2 = os.path.splitext(os.path.basename(nm2))[0]
        else:
            label2 = 'method 2'

    vib1 = vib.Vibrations(nm1, prop_prefix1)
    vib2 = vib.Vibrations(nm2, prop_prefix2)

    assert np.all(vib1.atoms.symbols == vib2.atoms.symbols)
    assert not np.all(vib1.frequencies == 0)
    assert not np.all(vib2.frequencies == 0)

    dot_product_matrix = pd.DataFrame()
    for i, (freq1, evec1) in enumerate(
            zip(vib1.frequencies, vib1.eigenvectors)):

        dot_product = [np.abs(np.dot(evec1, evec2)) for evec2 in
                       vib2.eigenvectors]
        dot_product_matrix[i] = dot_product

    N = 3 * len(vib1.atoms)
    if N > 30:
        figsize = (0.13 * N, 0.10 * N)
    else:
        figsize = (10, 8)

    tick_labels1 = [f'{f :3.1f}' for f in vib1.frequencies/invcm]
    tick_labels2 = [f'{f :3.1f}' for f in vib2.frequencies/invcm]

    if isinstance(cmap, str):
        if cmap in plt.colormaps():
            cmap = copy.copy(mpl.cm.get_cmap(cmap))
        else:
            cmap = cmap_in_alpha(cmap)

    if not adjust_color_range:
        norm = mpl.colors.Normalize(vmin=0 - tolerance, vmax=1 + tolerance)
        cmap.set_under(color='limegreen')
        cmap.set_over(color='dodgerblue')
    else:
        norm = None

    extend = 'neither'
    if max(dot_product_matrix.max()) > 1 + tolerance:
        extend = 'max'
    if min(dot_product_matrix.min()) < 0 - tolerance:
        if extend == 'max':
            extend = 'both'
        else:
            extend = 'min'

    fig, ax = plt.subplots(figsize=figsize)

    if arrange_by == 'order':

        hmap = ax.pcolormesh(dot_product_matrix, cmap=cmap, norm=norm, 
                            edgecolors='lightgrey', linewidth=0.02,  
                            linestyle=':')

        ax.set_xticks(np.arange(0.5, N, 1))
        ax.set_xticklabels(tick_labels1, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(0.5, N, 1))
        ax.set_yticklabels(tick_labels2, fontsize=6)

    elif arrange_by == 'frequency':

        scatter_data = {'xs': [], 'ys': [], 'colors': []}
        for x_freq, (x_idx, x_freq_dot_products) in zip(tick_labels1,
                                                dot_product_matrix.items()):
            for y_freq, (y_idx, dot_product) in zip(tick_labels2,
                                           x_freq_dot_products.items()):
                if dot_product < tolerance:
                    continue
                scatter_data['xs'].append(float(x_freq))
                scatter_data['ys'].append(float(y_freq))
                scatter_data['colors'].append(dot_product)

        hmap = ax.scatter(scatter_data['xs'], scatter_data['ys'],
                          c=scatter_data['colors'],
                          cmap=cmap, norm=norm, edgecolor='none')

        max_val = max(scatter_data['xs'] + scatter_data['ys'])
        ax.plot([0, max_val], [0, max_val], color='k', linewidth=0.4)

    cbar = plt.colorbar(hmap, extend=extend)
    cbar.set_label('Dot product', rotation=90)

    ax.set_xlabel(f'{label1} frequencies, cm$^{{-1}}$')
    ax.set_ylabel(f'{label2} frequencies, cm${{-1}}$')
    ax.set_title(f'Dot products between {label1} and {label2} eigenvectors',
                 fontsize=14)

    plt.savefig(fig_fname, dpi=300)
