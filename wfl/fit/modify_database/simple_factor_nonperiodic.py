from ase import Atoms

__default_config_type_settings = dict(dimer=[0.1, 0.5, None, None])


# fixme: the call args of this are taken to match the other one, we should generalise this
def modify(configs, overall_error_scale_factor=1.0, field_error_scale_factors=None, property_prefix='REF_'):
    """Modify the database (atoms objects) with a simple scaling of default_sigma values

    Parameters
    ----------
    configs: list(Atoms)
    overall_error_scale_factor: float, default=1.0
    field_error_scale_factors: dict
        this is a trick for now
    property_prefix

    Returns
    -------

    """

    if field_error_scale_factors is None:
        field_error_scale_factors = dict()

    # read the sigmas from file
    default_sigma = list_to_sigma_dict(field_error_scale_factors.get("default_sigma", [0.010, 0.150, None, None]))
    extra_space = field_error_scale_factors.get("extra_space", 20.)
    config_type_sigma = dict()
    for key, val in field_error_scale_factors.get("config_type_sigma", __default_config_type_settings).items():
        config_type_sigma[key] = list_to_sigma_dict(val)

    # loop over atoms and set sigma as default * factor
    for at in configs:
        modify_cell(at, extra_space=extra_space)

        # first deal with special nonperiodic configs
        if 'config_type' in at.info:
            if at.info["config_type"] == "isolated_atom":
                modify_with_factor(at, factor=1.0, energy_sigma=.0001, force_sigma=None, virial_sigma=None,
                                   hessian_sigma=None, property_prefix=property_prefix)
                continue
            elif at.info["config_type"] in config_type_sigma.keys():
                modify_with_factor(at, factor=overall_error_scale_factor, property_prefix=property_prefix,
                                   **config_type_sigma[at.info["config_type"]])
                continue
        # otherwise this just the
        modify_with_factor(at, factor=overall_error_scale_factor, property_prefix=property_prefix,
                           **default_sigma)


def list_to_sigma_dict(sigma_values):
    return {'energy_sigma': sigma_values[0],
            'force_sigma': sigma_values[1],
            'virial_sigma': sigma_values[2],
            'hessian_sigma': sigma_values[3]}


def modify_with_factor(at, factor=1.0, energy_sigma=None, force_sigma=None, virial_sigma=None, hessian_sigma=None,
                       property_prefix=None):
    """Set the kernel regularisation values and remove results if given but not

    Parameters
    ----------
    at
    factor: float, default=1.0
    energy_sigma
    force_sigma
    virial_sigma
    hessian_sigma
    property_prefix

    Returns
    -------
    None
    """

    if property_prefix is None:
        property_prefix = ""

    # cannot skip it

    if energy_sigma:
        at.info['energy_sigma'] = energy_sigma * factor
    else:
        try:
            del at.info[property_prefix + 'energy']
        except KeyError:
            pass

    # can skip forces
    if force_sigma:
        at.info['force_sigma'] = force_sigma * factor
    else:
        try:
            del at.arrays[property_prefix + 'forces']
        except KeyError:
            pass

    if virial_sigma:
        at.info['virial_sigma'] = virial_sigma * factor
    else:
        try:
            del at.arrays[property_prefix + 'virial']
        except KeyError:
            pass

    if hessian_sigma:
        at.info['hessian_sigma'] = hessian_sigma * factor
    else:
        try:
            del at.info[property_prefix + 'hessian']
        except KeyError:
            pass


def modify_cell(at: Atoms, extra_space=20.0):
    # set cell such big that atoms don't see one another
    pos = at.get_positions()
    cell = pos.max(axis=0) - pos.min(axis=0) + extra_space * 2
    at.cell = cell
