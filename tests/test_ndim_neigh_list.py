import numpy as np

from wfl.utils.ndim_neighbor_list import calc_list, calc_list_cells


def compare_manual_neighbor_list(positions, nearby_ranges, ii, jj, cartesian_distance=True):
    """compare neighbor list (presumably from linear scaling routine) to manual O(N^2) enumeration"""

    nn_pairs = set(zip(ii, jj))
    manual_pairs = set()
    for i, vi in enumerate(positions):
        for j, vj in enumerate(positions):
            dv = np.array(vi) - np.array(vj)
            if cartesian_distance:
                q = np.linalg.norm(dv / nearby_ranges)
            else:
                q = np.max(np.abs((dv / nearby_ranges)))
            if q < 1:
                manual_pairs |= {(i, j)}
    print('lengths', len(nn_pairs), len(manual_pairs))
    for p in nn_pairs:
        assert p in manual_pairs
    for p in manual_pairs:
        assert p in nn_pairs


def test_ndim_neighbor_list():
    np.random.seed(5)

    vals = np.random.normal(size=(1000, 2))

    print('fake data w/cartesian_distance=False')
    ii, jj = calc_list(vals, [0.05, 0.05], cartesian_distance=False)
    # compare neighbor list to manual routine
    compare_manual_neighbor_list(vals, [0.05, 0.05], ii, jj, cartesian_distance=False)

    print('fake data w/cartesian_distance default True')
    ii, jj = calc_list(vals, [0.05, 0.05])
    # compare neighbor list to manual routine
    compare_manual_neighbor_list(vals, [0.05, 0.05], ii, jj)

    print('comparing to cell list')
    # compare cell list to default routine
    ii_cells, jj_cells = calc_list_cells(vals, [0.05, 0.05])

    print('avg neighb #', len(ii) / len(vals))
    print('cell avg neighb #', len(ii_cells) / len(vals))

    for i in range(len(vals)):
        ii_inds = np.where(ii == i)[0]
        assert set(ii[ii_inds]) == set(ii_cells[ii_inds])
        assert set(jj[ii_inds]) == set(jj_cells[ii_inds])
