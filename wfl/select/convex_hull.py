import sys

from wfl.select.selection_space import composition_space_Zs, composition_space_coord
from wfl.utils.convex_hull import find_hull


def select(inputs, outputs, info_field, Zs=None, verbose=False):
    if outputs.all_written():
        sys.stderr.write('Returning from {__name__} since output is done\n')
        return outputs.to_ConfigSet()

    if Zs is None:
        Zs = composition_space_Zs(inputs)

    positions = []
    avail_inds = {}
    for at_i, at in enumerate(inputs):
        if info_field in at.info:
            positions.append(composition_space_coord(at, ['_V', '_x', info_field], Zs))
            avail_inds[at_i] = len(positions) - 1

    # convert to set for faster checking (O(1)?) of "in" below
    _, indices, _, simplices = find_hull(positions)
    selected_indices = set(indices)
    if verbose:
        for s_i, s in enumerate(simplices):
            print('arb_polyhedra -name {} -indices {}'.format(s_i, ' '.join([str(i) for i in s])))

    for at_i, at in enumerate(inputs):
        try:
            if avail_inds[at_i] in selected_indices:
                outputs.store(at)
        except KeyError:
            # skip configs that are not in avail_inds
            pass

    outputs.close()
    return outputs.to_ConfigSet()
