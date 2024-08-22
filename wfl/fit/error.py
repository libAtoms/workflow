import warnings
import re

import pandas as pd
import numpy as np

from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap


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
        diffs: dict with list of differences for each category and property (signed for scalar
            properties, norms for vectors)
        parity: dict with "ref" and "calc" keys, each containing list of property values for
            each category and property, for parity plots
    """

    # default properties
    if ((config_properties is None or len(config_properties) == 0) and
        (atom_properties is None or len(atom_properties) == 0)):
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
    all_parity = {"ref": {}, "calc": {}}
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
            else:  # atom_properties
                if per_atom:
                    raise ValueError("/atom only possible in config_properties")
                data = at.arrays

            # grab data
            ref_quant = data.get(ref_property_prefix + prop_use)
            calc_quant = data.get(calc_property_prefix + prop_use)
            if ref_quant is None or calc_quant is None:
                # warn if data is missing by reporting summary at the very end
                if prop not in missed_prop_counter:
                    missed_prop_counter[prop] = 0
                missed_prop_counter[prop] += 1

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
            else:  # atom_properties
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
                if diff.shape[1] > 1:
                    # compute norm along vector components
                    diff = np.linalg.norm(diff, axis=1)
                if not per_component and selected_ref_quant.shape[1] > 1:
                    selected_ref_quant = np.linalg.norm(selected_ref_quant, axis=1)
                    selected_calc_quant = np.linalg.norm(selected_calc_quant, axis=1)

                _dict_add([all_diffs, all_weights,            all_parity["ref"],  all_parity["calc"]],
                          [diff,      _promote(weight, diff), selected_ref_quant, selected_calc_quant],
                          at_category, prop + atom_split_index_label)

    if len(missed_prop_counter.keys()) > 0:
        for missed_prop, count in missed_prop_counter.items():
            if count == len(list(inputs)):
                raise RuntimeError(f"Missing reference ({ref_property_prefix}) or calculated ({calc_property_prefix}) "
                                    f"property '{missed_prop}' in all of the configs. Is the spelling correct? ")
            else:
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

            all_errors[prop][cat] = {'RMSE': RMSE, 'MAE': MAE, 'count': num}

    # if len(all_diffs) == 0:
    #     raise RuntimeError(f"No values were found to calculate error from."\
    #                        f"This might be because of misspelled property prefixes ({calc_property_prefix} or {ref_property_prefix})")

    return all_errors, all_diffs, all_parity


def value_error_scatter(all_errors, all_diffs, all_parity, output, properties=None,
                        ref_property_prefix="reference ", calc_property_prefix="calculated ", error_type="RMSE",
                        plot_parity=True, plot_error=True, cmap=None, units_dict=None):
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
    calc_property_prefix: str, default "calculated "
        prefix for reference property labels
    error_type: str in ["RMSE, "MAE"], default "RMSE"
        root-mean-square or maximum-absolute error
    cmap: str, default None
        colormap name to use. If None use default based on number of categories.
    units_dict: dict, default None
        dictionary with units for non-default properties. Example:
        {"energy": {"parity": ("eV", 1.0), "error": ("meV", 1.0e3)}},
        Where each tuple containes units lable and conversion factor from ASE-default units.
    """

    assert error_type in ["RMSE", "MAE"], f"'error_type' must be 'RMSE' or 'MAE', not {error_type}."

    num_rows = sum([plot_parity, plot_error])  # one for parity, one for errors
    num_columns = len(all_errors.keys())  # one column per property
    side = 4.5
    fig = Figure(figsize=(side * num_columns, side * num_rows))
    gs = fig.add_gridspec(num_rows, num_columns, wspace=0.25, hspace=0.25)

    if properties is None:
        properties = list(all_errors.keys())

    num_cat = len(list(all_errors[properties[0]].keys()))

    # set up colormap
    if cmap is not None:
        cmap = get_cmap(cmap)
        colors = [cmap(idx) for idx in np.linspace(0, 1, num_cat)]
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

            units_factor = select_units(prop, "error", units_dict=units_dict)
            label = f'{category}: {errors[category][error_type] * units_factor[1]:.2f} {units_factor[0]}'
            if ax_parity is not None:
                ax_parity.scatter(ref_vals[category], pred_vals[category], label=label,
                                  edgecolors=color, facecolors='none')
            if ax_error is not None:
                ax_error.scatter(ref_vals[category], np.array(differences[category]) * units_factor[1],
                                 edgecolors=color, facecolors='none')

        if ax_parity is not None:
            _annotate_parity_plot(ax_parity, prop, ref_property_prefix, calc_property_prefix,
                                  show_legend, error_type, units_dict)
        if ax_error is not None:
            _annotate_error_plot(ax_error, prop, ref_property_prefix, calc_property_prefix, units_dict)

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


def select_units(prop, plt_type, units_dict=None):
    """
    select unit labels and conversion factors from ASE default for each property


    Parameters
    ----------
    prop: str
        name of property. By default upported "energy", "atomization_energy", "forces" and "virial",
        in combination with "/atom", "/comp", "/Z" as appropriate.
    plt_type: str in ["error", "parity"]
        type of plot
    units_dict: dict, default None
        dictionary with units for non-default properties. Example:
        {"energy": {"parity": ("eV", 1.0), "error": ("meV", 1.0e3)}},
        Where each tuple containes units lable and conversion factor from ASE-default units.


    Returns
    -------
    property_label: str
    conversion_factor: float to multiply raw quantity by
    """

    use_units_dict = {
        "energy": {"parity": ("eV", 1.0), "error": ("meV", 1.0e3)},
        "energy/atom": {"parity": ("eV/at", 1.0), "error": ("meV/at", 1.0e3)},
        "forces": {"parity": ("eV/Å", 1.0), "error": ("meV/Å", 1.0e3)},
        "virial": {"parity": ("eV", 1.0), "error": ("meV", 1.0e3)},
        "virial/atom": {"parity": ("eV/at", 1.0), "error": ("meV/at", 1.0e3)},
        "stress": {"parity": ("GPa", 1.0), "error": ("MPa", 1.0e3)},
    }
    if units_dict is None:
        units_dict = {}
    use_units_dict.update(units_dict)

    prop = re.sub(r"/comp\b", "", prop)
    if "forces" in prop:
        prop = re.sub(r"/Z_\d+\b", "", prop)

    if "energy" in prop:
        # also support `atomization_energy`
        prop = re.sub(r"atomization_energy", "energy", prop)

    if prop not in use_units_dict or plt_type not in use_units_dict[prop]:
        raise KeyError(f"Unknown property ({prop}) or plot type ({plt_type}).")

    return use_units_dict[prop][plt_type]


def _annotate_parity_plot(ax, property, ref_property_prefix, calc_property_prefix, show_legend, error_type, units_dict=None):
    if show_legend:
        ax.legend(title=f"{error_type} per category")
    ax.set_title(f"{property} ")
    ax.set_xlabel(f"{ref_property_prefix}{property}, {select_units(property, 'parity', units_dict=units_dict)[0]}")
    ax.set_ylabel(f"{calc_property_prefix}{property}, {select_units(property, 'parity', units_dict=units_dict)[0]}")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lims = (min([xmin, ymin]) - 0.1, max([xmax, ymax]) + 0.1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, c='k', linewidth=0.8)

    ax.grid(color='lightgrey', ls=':')


def _annotate_error_plot(ax, property, ref_property_prefix, calc_property_prefix, units_dict=None):
    ax.set_title(f"{property} error")
    ax.set_xlabel(f"{ref_property_prefix}{property}, {select_units(property, 'parity')[0]}")
    ax.set_ylabel(f"{calc_property_prefix}{property} error, {select_units(property, 'error')[0]}")
    ax.grid(color='lightgrey', ls=':')
    ax.set_yscale('log')


def errors_dumps(errors, error_type="RMSE", precision=2):
    """converts errors dictionary to dataframe and prints nicely.

    Parameters
    ----------
    errors: dict
        Dictionary returned by wfl.fit.error.calculate()
    error_type: str in ["RMSE, "MAE"], default "RMSE"
        root-mean-square or maximum-absolute error
    precision: int, default 2
        Float precision when printing
    """

    df = errors_to_dataframe(errors, error_type=error_type)
    df_str = df.to_string(
        max_rows=None,
        max_cols=None,
        float_format="{{:.{:d}f}}".format(precision).format,
        sparsify=False
    )
    return df_str


def errors_to_dataframe(errors, error_type="RMSE", units_dict=None):
    """converts errors dictionary to dataframe with properties as columns
    and categories as rows

    Parameters
    ----------
    errors: dict
        Dictionary returned by wfl.fit.error.calculate()
    error_type: str in ["RMSE, "MAE"], default "RMSE"
        root-mean-square or maximum-absolute error
    units_dict: dict, None
        dictionary with units for non-default properties. Example:
        {"energy": {"parity": ("eV", 1.0), "error": ("meV", 1.0e3)}},
        Where each tuple containes units lable and conversion factor from ASE-default units.
    """
    # errors keys
    # property : category : RMSE/MAE/count

    # store error_type value and counts in their own columns
    df_errors = {}
    for prop in errors:
        units_factor = select_units(prop, 'error', units_dict=units_dict)

        prop_header = re.sub(r'^energy\b', 'E', prop)
        prop_header = re.sub(r'^forces\b', 'F', prop_header)
        prop_header = re.sub(r'^virial\b', 'V', prop_header)
        prop_header = re.sub(r'/atom\b', '/a', prop_header)
        prop_header = re.sub(r'/comp\b', '/c', prop_header)

        # tuple keys become pd.MultiIndex and therefore grouped two-row display header
        prop_val_key = (prop_header, units_factor[0])
        prop_count_key = (prop_header, "#")

        df_errors[prop_val_key] = {}
        df_errors[prop_count_key] = {}
        for cat in errors[prop]:
            df_errors[prop_val_key][cat] = errors[prop][cat][error_type] * units_factor[1]
            df_errors[prop_count_key][cat] = int(errors[prop][cat]["count"])

    # fill in 0 for missing counts
    all_cats = set([cat for prop in errors for cat in errors[prop]])
    for prop in df_errors:
        if "#" not in prop[1]:
            continue
        for cat in all_cats:
            if cat not in df_errors[prop]:
                df_errors[prop][cat] = 0

    df = pd.DataFrame.from_dict(df_errors)

    # sort rows alphabetically
    new_rows = [row for row in df.index if row != "_ALL_"]
    new_rows = natural_sort(new_rows) + ["_ALL_"]

    df.sort_index(key=lambda index: pd.Index([new_rows.index(i) for i in index]), inplace=True)

    return df

def natural_sort(l):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
