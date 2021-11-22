import sys

import numpy as np

from wfl.utils.vol_composition_space import composition_space_Zs, composition_space_coord


def minima_among_neighbors(positions, ranges, values, cartesian_distance=True):
    assert positions.shape[1] == len(ranges)
    assert len(positions) == len(values)

    minima = []
    for i in range(len(positions)):
        dval = (positions[:] - positions[i]) / ranges
        if cartesian_distance:
            nearby = np.where(np.linalg.norm(dval, axis=1) < 1.0)[0]
        else:
            nearby = np.where(np.all(np.abs(dval) < 1.0, axis=1))[0]
        minima.append(min(values[nearby]))
    return minima


def compare_manual_minima(i, j, positions, nn_minima):
    """compare nearby minima (presumably from efficient np.reduceat) to manual enumeration"""
    for ii in set(i):
        js = j[np.where(i == ii)[0]]
        if np.min(positions[js]) != nn_minima[ii]:
            print('mismatch', ii, np.min(positions[js]), nn_minima[ii])


# should this be refactored so it can use pipeline.iterable_loop?
def val_relative_to_nearby_composition_volume_min(inputs, outputs, vol_range, compos_range, info_field_in,
                                                  info_field_out,
                                                  Zs=None, per_atom=True):
    """compute difference between some value to corresponding values for configurations
    that are nearby in compositions/volume space

    Parameters
    ----------
    inputs: ConfigSet_in
        input configurations
    outputs: ConfigSet_out
        corresponding place for output configs
    vol_range: float
        cutoff range for "nearby" in cell volume/atom [we should define what to do about nonperiodic systems]
    compos_range: float
        cutoff range for "nearby" in composition (fractional, i.e. 0.0-1.0)
    info_field_in: str
        Atoms.info field containing quantity to be subtracted relative to "nearby" configs
    info_field_out: str
        Atoms.info field to store value differences in
    Zs: list(int), default None
        Zs that defined the composition space, if None get from inputs
    per_atom: bool, default True
        apply calculations to per-atom quantities, i.e. Atoms.info[info_field_in] / len(atoms)

    Returns
    -------
    ConfigSet_in
        ConfigSet_in pointing to configurations with the saved relative value field
    """
    if outputs.is_done():
        sys.stderr.write(f'Returning from {__name__} since output is done\n')
        return outputs.to_ConfigSet_in()

    if Zs is None:
        Zs = composition_space_Zs(inputs)

    positions = []
    values = []
    for at_i, at in enumerate(inputs):
        if at_i % 1000 == 1000 - 1:
            if at_i % 10000 == 10000 - 1:
                sys.stderr.write('{}'.format((at_i + 1) // 10000 % 10))
            else:
                sys.stderr.write('.')
            if at_i % 100000 == 100000 - 1:
                sys.stderr.write('\n')
            sys.stderr.flush()
        positions.append(composition_space_coord(at, ['_V', '_x'], Zs))
        v = at.info[info_field_in]
        if per_atom:
            v /= len(at)
        values.append(v)
    positions = np.asarray(positions)
    values = np.asarray(values)

    nearby_ranges = [vol_range] + [compos_range] * (len(Zs) - 1)
    minima = minima_among_neighbors(positions, nearby_ranges, values, cartesian_distance=False)

    for at, minimum in zip(inputs, minima):
        v = at.info[info_field_in]
        if per_atom:
            v /= len(at)
        at.info[info_field_out] = v - minimum
        outputs.write(at, from_input_file=inputs.get_current_input_file())
    outputs.end_write()

    return outputs.to_ConfigSet_in()
