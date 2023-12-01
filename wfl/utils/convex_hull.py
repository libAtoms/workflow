import numpy as np

from scipy.spatial import ConvexHull


# manually remove indices with range < 1e-10.  Should be able to use qhull QbN:0BN:0,
# where N is index number, but gives seg fault
def find_hull(ps, below=True):
    """find convex hull of set of points

    Parameters
    ----------
    ps: ndarray(n_ps, n_dim)
        array of positions in arbitrary dim space
    below: bool, default True
        only return parts of hull that are "below" all of the other points, i.e. lowest values of n_dim-1 component (assumed to be energy)

    Returns
    -------
        points: ndarray(n_hull_ps, n_dim)
            points on hull
        indices: list(int)
            list of indices of hull points
        equations: ConvexHull.equations
            equations representing convex hull simplices
        simplices: list(list(int))
            list of indices in each hull simplex
    """

    if not isinstance(ps, np.ndarray):
        ps = np.array(ps)
    # check for indices with zero range, not including final (energy)
    non_degenerate_inds = []
    for i in range(len(ps[0]) - 1):
        if np.max(ps[:, i]) - np.min(ps[:, i]) > 1.0e-10:
            non_degenerate_inds.append(i)
    non_degenerate_inds += [len(ps[0]) - 1]

    # create points with indices dropped
    if len(non_degenerate_inds) != len(ps[0]):
        ps_clean = np.array(ps)[:, non_degenerate_inds]
    else:
        ps_clean = ps

    if ps_clean.shape[0] < ps_clean.shape[1]:
        raise RuntimeError(
            "Need at least as many points {} as non-degenerate dimensions {} to make a convex hull".format(
                ps_clean.shape[0], ps_clean.shape[1]))

    # find convex hull
    hull = ConvexHull(ps_clean)

    indices = set()
    equations = []
    simplices = []
    for (simplex, equation) in zip(hull.simplices, hull.equations):
        # select equations for faces that define only _below_ part of convex hull
        if not below or equation[-2] < 0:
            indices |= set(simplex)
            equations.append(equation)
            simplices.append(simplex)

    # add indices back into equations
    if len(non_degenerate_inds) != len(ps[0]):
        equations = np.array(equations)
        eqns_out = np.zeros((equations.shape[0], ps.shape[1] + 1))

        eqns_out[:, non_degenerate_inds] = equations[:, :-1]
        eqns_out[:, -1] = equations[:, -1]
    else:
        eqns_out = equations

    return [ps[i] for i in indices], list(indices), eqns_out, simplices


def vertical_dist_from_hull(equations, p):
    min_dist = None
    for eq in equations:
        v = eq[:-1]
        offset = eq[-1]
        # v.(p - x yhat) + offset = 0
        # v.p - x v.yhat = -offset
        # x = -(-offset - v.p) / (v.yhat)
        d = (offset + np.dot(v, p)) / v[-1]
        min_dist = d if (min_dist is None or d < min_dist) else min_dist
    return min_dist
