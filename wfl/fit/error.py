import warnings
import re

import numpy as np
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap



def calc(inputs, calc_property_prefix, ref_property_prefix,
         config_properties=None, atom_properties=None, category_keys="config_type",
         weight_property=None):
    """calculate error for calculation results relative to stored reference values

    Parameters
    ----------
    inputs: iterable(Atoms)
        input configurations with reference and calculated values in Atoms.info/arrays
    calc_property_prefix: str
        prefix to info/array keys for calculated properties
    ref_property_prefix: str
        prefix to info/array keys for reference properties
    config_properties: list(str), default ["energy/atom", "virial/atom/comp"]
        list of ``Atoms.info`` calculated properties (to be prefixed by ``ref_property_prefix``
        or ``calc_property_prefix``) to compute error for.  ``virial`` will be reconstructed from ``stress``.
        Properties can end with ``/atom`` or ``/comp`` for different components being counted separately.
        Default only used if neither ``config_properties`` nor ``atom_properties`` is present.
    atom_properties: list(str), default ["forces"]
        list of `Atoms.arrays` calculated properties (to be prefixed by ``ref_property_prefix``
        or ``calc_property_prefix``) to compute error For.  Properties can end with ``/comp``
        for different components being computed separately, and ``/Z`` for different atomic numbers
        to be assigned to different categories.  Default only used if neither ``config_properties``
        nor ``atom_properties`` is present.
    category_keys: str / list(str), default "config_type"
        results will be averaged by category, defined by a string containing the values of these
        keys in Atoms.info, in addition to overall average _ALL_ category.
    weight_property: str, optional
        if present, Atoms.info key for weights to apply to RMSE calculation

    Returns
    -------
        errors: dict of RMSE and MAE for each category and property
        diffs: dict with list of differences for each category and property
        parity: dict with "ref" and "calc" keys, each containing list of property values for
            each category and property, for parity plots
    """

    # default properties
    if config_properties is None and atom_properties is None:
        config_properties = ["energy/atom", "virial/atom/comp"]
        atom_properties = ["forces"]
    if config_properties is None:
        config_properties = []
    if atom_properties is None:
        atom_properties = []

    # clean up category_keys
    if isinstance(category_keys, str):
        category_keys = [category_keys]
    elif category_keys is None:
        category_keys = [None]

    def _reshape_normalize(quant, prop, atoms, per_atom):
        """reshape and normalize quantity so its error can be calculated cleanly

        Parameters
        ----------
        quant: scalar number or array of numbers
            quantity whose error will be computed.  Can be scalar or arbitrary shape
            array (if per-config) or array with leading dimension len(at) (if per-atom)
        prop: str
            name of property
        atoms: Atoms
            corresponding atoms object
        per_atom: bool
            quantity should be divided by len(atoms)

        Returns
        -------
        quant: 2-d array containing reshaped quantity, with leading dimension 1 for per-config
            or len(atoms) for per-atom
        """
        # convert scalars or lists into arrays
        quant = np.asarray(quant)

        # Reshape to 2-d, with leading dimension 1 for per-config, and len(atoms) for per-atom.
        # This is the right shape to work with later flattening for per-property and norm calculation
        # for vector property differences.
        if prop in config_properties:
            quant = quant.reshape((1, -1))
        else:
            # flatten all except per-atom dimension
            quant = quant.reshape((len(atoms), -1))

        if per_atom:
            quant /= len(atoms)

        return quant

    # compute diffs and store in all_diffs, and weights in all_weights
    all_diffs = {}
    all_parity = { "ref": {}, "calc": {} }
    all_weights = {}

    missed_prop_counter = {}

    for at_idx, at in enumerate(inputs):
        # turn category keys into a single string for dict key
        at_category = " / ".join([str(at.info.get(k)) for k in category_keys])
        weight = at.info.get(weight_property, 1.0)

        for prop in config_properties + atom_properties:
            prop_use = prop

            # parse (and remove) "/..." suffixes
            per_atom = re.search(r"/atom\b", prop_use)
            prop_use = re.sub(r"/atom\b", "", prop_use)
            per_component = re.search(r"/comp\b", prop)
            prop_use = re.sub(r"/comp\b", "", prop_use)
            by_species = re.search(r"/Z\b", prop_use)
            prop_use = re.sub(r"/Z\b", "", prop_use)

            # possibly reconstruct virial later
            virial_from_stress = False
            if prop_use == "virial":
                prop_use = "stress"
                virial_from_stress = True

            # select dict and check for inconsistencies
            if prop in config_properties:
                if by_species:
                    raise ValueError("/Z only possible in atom_properties")
                data = at.info
            else: # atom_properties
                if per_atom:
                    raise ValueError("/atom only possible in config_properties")
                data = at.arrays

            # grab data
            ref_quant = data.get(ref_property_prefix + prop_use)
            calc_quant = data.get(calc_property_prefix + prop_use)
            if ref_quant is None or calc_quant is None:
                # warn if data is missing by reporting summary at the very end
                if prop_use not in missed_prop_counter:
                    missed_prop_counter[prop_use] = 0
                missed_prop_counter[prop_use] += 1

                continue

            if virial_from_stress:
                # ref quant was actually stress, automatically convert
                ref_quant *= -at.get_volume()
                calc_quant *= -at.get_volume()

            # make everything into an appropriately shaped and normalized array
            ref_quant = _reshape_normalize(ref_quant, prop, at, per_atom)
            calc_quant = _reshape_normalize(calc_quant, prop, at, per_atom)

            if prop in config_properties or not by_species:
                # If quantities are not being split up by species, make a
                # group that includes all atoms, so loop below that separates out
                # things by Z will lump them all together.
                atom_split_indices = np.asarray(range(len(ref_quant)))
                atom_split_groups = [(atom_split_indices, "")]
            else: # atom_properties
                # Make separate groups for each atomic number Z, so their errors
                # can be tabulated separately.
                Zs = at.numbers
                atom_split_indices = np.asarray(Zs)
                atom_split_groups = [(Z, f"_{Z}") for Z in sorted(set(Zs))]

            # Loop over groups that errors should be split up by within each configuration.
            # Only atomic number Z implemented so far (see above).
            for atom_split_index_val, atom_split_index_label in atom_split_groups:
                # use only subset of quantities that are relevant to this subset of atoms,
                # normally either all atoms or the ones with one particular atomic number
                selected_ref_quant = ref_quant[atom_split_indices == atom_split_index_val]
                selected_calc_quant = calc_quant[atom_split_indices == atom_split_index_val]

                if per_component:
                    # if per component, flatten all vector component so each is counted separately
                    selected_ref_quant = selected_ref_quant.reshape((-1, 1))
                    selected_calc_quant = selected_calc_quant.reshape((-1, 1))

                diff = selected_calc_quant - selected_ref_quant

                if len(diff.shape) != 2:
                    raise RuntimeError(f"Should never have diff.shape={diff.shape} with dim != 2 (prop {prop + atom_split_index_label})")
                # compute norm along vector components
                diff = np.linalg.norm(diff, axis=1)
                if not per_component:
                    selected_ref_quant = np.linalg.norm(selected_ref_quant, axis=1)
                    selected_calc_quant = np.linalg.norm(selected_calc_quant, axis=1)


                _dict_add([all_diffs, all_weights,            all_parity["ref"],   all_parity["calc"]], 
                          [diff,      _promote(weight, diff), selected_ref_quant,           selected_calc_quant        ],
                          at_category, prop + atom_split_index_label)

    if len(missed_prop_counter.keys()) > 0:
        for missed_prop, count in missed_prop_counter.items():
            warnings.warn(f"Missing reference or calculated property '{missed_prop}', {count} times")
            

    all_errors = {}
    for prop in all_diffs:
        all_errors[prop] = {}
        for cat in all_diffs[prop]:
            diffs = np.asarray(all_diffs[prop][cat])
            weights = np.asarray(all_weights[prop][cat])

            RMSE = np.sqrt(np.sum((diffs ** 2) * weights) / np.sum(weights))
            MAE = np.sum(np.abs(diffs) * weights) / np.sum(weights)
            num = len(diffs)

            all_errors[prop][cat] = {'RMSE': RMSE, 'MAE': MAE, 'count' : num}

    return all_errors, all_diffs, all_parity


def value_error_scatter(all_errors, all_diffs, all_parity, output, properties=None,
                        ref_property_prefix="reference ", calc_property_prefix="calculated ", error_type="RMSE",
                        plot_parity=True, plot_error=True, cmap_name_bins=None):
    """generate parity plot (calculated values vs. reference values) and/or scatterplot of 
    errors vs. values

    Parameters
    ----------
    all_errors: dict
        dict of errors returned by error.calc (first returned item)
    all_diffs: dict
        dict of property differences returned by error.calc (second returned item)
    all_parity: dict
        dict of property values for parity returned by error.calc (third returned item)
    output: str
        output filename
    properties: list(str), default None
        properties to plot, if None plot all
    ref_property_prefix: str, default "reference "
        prefix for reference property labels
    ref_property_prefix: str, default "calculated "
        prefix for reference property labels
    error_type: str, default "RMSE"
        type of error matching key in all_errors dict
    cmap_name_bins: tuple(str, int), default None
        colormap to use and number of bins. If None use number of categories based default
    """

    assert error_type in ["RMSE", "MAE"], f"'error_type' must be 'RMSE' or 'MAE', not {error_type}."

    num_rows = sum([plot_parity, plot_error])   # one for parity, one for errors
    num_columns = len(all_errors.keys()) # one column per property
    side = 4.5
    fig = Figure(figsize=(side * num_columns, side * num_rows))
    gs = fig.add_gridspec(num_rows, num_columns, wspace=0.25, hspace=0.25)

    if properties is None:
        properties = list(all_errors.keys())

    num_cat = len(list(all_errors[properties[0]].keys()))

    # set up colormap
    if cmap_name_bins is not None:
        assert len(cmap_name_bins) == 2
        cmap = get_cmap(cmap_name_bins[0])
        colors = [cmap(idx) for idx in np.linspace(0, 1, cmap_name_bins[1])]
    else:
        if num_cat < 11:
            cmap = get_cmap('tab10')
            colors = [cmap(idx) for idx in np.linspace(0, 1, 10)]
        else:
            cmap = get_cmap("hsv")
            colors = [cmap(idx) for idx in np.linspace(0, 1, num_cat)]
    show_legend = num_cat <= 10

    sel_diffs = [all_diffs[prop] for prop in properties]
    sel_parity_ref = [all_parity["ref"][prop] for prop in properties]
    sel_parity_calc = [all_parity["calc"][prop] for prop in properties]
    sel_errors = [all_errors[prop] for prop in properties]

    plot_iter = zip(properties, sel_diffs, sel_parity_ref, sel_parity_calc, sel_errors)
    for prop_idx, (prop, differences, ref_vals, pred_vals, errors) in enumerate(plot_iter):

        gs_idx = 0
        if plot_parity:
            ax_parity = fig.add_subplot(gs[gs_idx, prop_idx])
            gs_idx += 1
        else:
            ax_parity = None
        if plot_error:
            ax_error = fig.add_subplot(gs[gs_idx, prop_idx], sharex=ax_parity)
        else:
            ax_error = None

        for cat_idx, category in enumerate(differences.keys()):
            if category == "_ALL_":
                continue
            
            color = colors[cat_idx]
                
            label = f'{category}: {errors[category][error_type]*1e3:.2f} {select_units(prop, "error")}'
            if ax_parity is not None:
                ax_parity.scatter(ref_vals[category], pred_vals[category], label=label,
                                  edgecolors=color, facecolors='none')
            if ax_error is not None:
                ax_error.scatter(ref_vals[category], np.array(differences[category])*1e3,
                                 edgecolors=color, facecolors='none')

        if ax_parity is not None:
            _annotate_parity_plot(ax_parity, prop, ref_property_prefix, calc_property_prefix,
                                  show_legend, error_type)
        if ax_error is not None:
            _annotate_error_plot(ax_error, prop, ref_property_prefix, calc_property_prefix)

    fig.savefig(output, dpi=300, bbox_inches="tight")


def _promote(weight, val):
    try:
        return weight * np.ones(val.shape)
    except AttributeError:
        return weight


def _dict_add(dicts, values, at_category, prop):
    if at_category == "_ALL_":
        cats = [at_category]
    else:
        cats = [at_category, "_ALL_"]

    for d, v in zip(dicts, values):
        for cat in cats:
            if prop not in d:
                d[prop] = {}
            if cat not in d[prop]:
                d[prop][cat] = []
            d[prop][cat].extend(v)


def select_units(prop, plt_type):

    units_dict = {
    "energy/atom": {"parity": "eV/at", "error": "meV/at"},
    "energy": {"parity": "eV", "error": "meV"},
    "forces": {"parity": "eV/Å", "error": "meV/Å"},
    "virial": {"parity": "eV", "error": "meV"},
    "virial/atom": {"parity": "eV/at", "error": "meV/at"}
    }
    if "virial" in prop:
        prop = re.sub(r"/comp\b", "", prop)
    if "forces" in prop:
        prop = "forces"
    try:
        return units_dict[prop][plt_type]
    except KeyError:
        return "unknown units"


def _annotate_parity_plot(ax, property, ref_property_prefix, calc_property_prefix, show_legend, error_type):
    if show_legend:
        ax.legend(title=f"{error_type} per category")
    ax.set_title(f"{property} ")
    ax.set_xlabel(f"{ref_property_prefix}{property}, {select_units(property, 'parity')}")
    ax.set_ylabel(f"{calc_property_prefix}{property}, {select_units(property, 'parity')}")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lims = (min([xmin, ymin])-0.1, max([xmax, ymax])+0.1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, c='k', linewidth=0.8)

    ax.grid(color='lightgrey', ls=':')


def _annotate_error_plot(ax, property, ref_property_prefix, calc_property_prefix):
    ax.set_title(f"{property} error")
    ax.set_xlabel(f"{ref_property_prefix}{property}, {select_units(property, 'parity')}")
    ax.set_ylabel(f"{calc_property_prefix}{property} error, {select_units(property, 'error')}") 
    ax.grid(color='lightgrey', ls=':')
    ax.set_yscale('log')


