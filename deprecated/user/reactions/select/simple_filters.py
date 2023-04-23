# STUB deprecated from wfl.select.simple_filters
def by_energy(inputs, outputs, lower_limit, upper_limit, energy_parameter_name=None, e0=None):
    """Filter by binding energy

    Parameters
    ----------
    inputs: ConfigSet
        source configurations
    outputs: OutputSpec
        output configurations
    lower_limit: float / None
        lower energy limit for binding energy, None is -inf
    upper_limit: float / None
        upper energy limit for binging energy, None is +inf
    energy_parameter_name: str / None, default None
        parameter name to use for energy, if None then atoms.get_potential_energy() is used
    e0 : dict / None
        energy of isolated atoms, to use for binding energy calculation, with chemical symbols as keys
        None triggers all zero

    Returns
    -------
    ConfigSet pointing to selected configurations

    """

    if lower_limit is None:
        lower_limit = - np.inf

    if upper_limit is None:
        upper_limit = np.inf

    def get_energy(at: Atoms):
        if e0 is None:
            shift = 0.
        else:
            shift = np.sum([e0[symbol] for symbol in at.get_chemical_symbols()])

        if energy_parameter_name is None:
            return (at.get_potential_energy() - shift) / len(at)
        else:
            return (at.info.get(energy_parameter_name) - shift) / len(at)

    if outputs.is_done():
        sys.stderr.write(f'Returning before by_energy since output is done\n')
        return outputs.to_ConfigSet()

    selected_any = False
    for atoms in inputs:
        if lower_limit < get_energy(atoms) < upper_limit:
            outputs.write(atoms)
            selected_any = True

    outputs.end_write()
    if selected_any:
        return outputs.to_ConfigSet()
    else:
        return ConfigSet(input_configs=[])

