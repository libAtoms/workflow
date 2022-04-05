import sys
import warnings

import numpy as np

from wfl.utils.convex_hull import find_hull, vertical_dist_from_hull
from wfl.utils.vol_composition_space import composition_space_Zs, composition_space_coord


def piecewise_linear(x, vals):
    i = np.searchsorted([v[0] for v in vals], x)
    if i == 0:  # below first
        return np.array(vals[0][1])
    elif i == len(vals):  # above last
        return np.array(vals[-1][1])
    else:
        f0 = (vals[i][0] - x) / (vals[i][0] - vals[i - 1][0])
        return f0 * np.array(vals[i - 1][1]) + (1.0 - f0) * np.array(vals[i][1])


# max energy 20
# isolated_atom uses 0.0001:0.1:0.1:0.1
# dimer uses 0.1:0.5:1.0:1.0
# other configs:
#     energy_sigma constant 0.001 eV up to 0.2 eV, then linear ramp to 0.1 eV by 1.0 eV, then constant above that
#     energy difference measured relative to convex hull
#     other sigmas are set by sigma_F = sqrt(sigma_E), sigma_V = sigma_H = 2.0*sqrt(sigma_E)

def modify(configs, overall_error_scale_factor=1.0, field_error_scale_factors=None, property_prefix='REF_'):
    if field_error_scale_factors is None:
        field_error_scale_factors = {}
    property_keys = ['energy', 'forces', 'virial', 'hessian']
    Zs = composition_space_Zs(configs)

    # create convex hulls, separately for each value of at.info["gap_rss_group"]
    group_points = {}
    for group in set([at.info.get("gap_rss_group", None) for at in configs if all(at.pbc)]):
        group_points[group] = []
    for at_i, at in enumerate(configs):
        # modify will depend on getting values from at.info and at.arrays, so disable calc
        # hope that quantities are in there
        at.calc = None

        # skip configs that were ignored earlier because their energy was too high (> 20) before we try to
        #  use their energy and raise an exception
        if 'ignored_' + property_prefix + 'energy' in at.info:
            continue
        # skip configs that are nonperiodic, since their volume is meaningless and we're doing a convex hull in
        # V+x space
        if not all(at.pbc):
            continue
        # warning about configs that are missing the fitting energy and cannot have their sigma set
        if property_prefix + 'energy' not in at.info:
            warnings.warn(
                f'While modifying sigma, skipping configuration # {at_i} that does not have "{property_prefix}_energy" field')
            continue
        # this will raise an exception if there's a problem
        convex_hull_p = composition_space_coord(at, ["_V", "_x", property_prefix + 'energy'], Zs)
        at.info["fit_sigma_convex_hull_p"] = convex_hull_p
        group_points[at.info.get("gap_rss_group")].append(convex_hull_p)
    hull = {}
    eqns = {}
    for (group, points) in group_points.items():
        points = np.array(points)
        if points.shape[0] < points.shape[1]:
            raise RuntimeError(
                "Need at least as many points {} as dimensions {} to make a convex hull for group {}".format(
                    points.shape[0], points.shape[1], group))
        (hull[group], _, eqns[group], _) = find_hull(points)

    # loop over atoms and set sigmas by distance above (E,V) convex hull
    for at in configs:

        # first deal with special nonperiodic configs
        if 'config_type' in at.info and at.info['config_type'] == 'isolated_atom':
            # print("got single atom")
            at.info['energy_sigma'] = 0.0001
            try:
                del at.arrays[property_prefix + 'forces']
                del at.info[property_prefix + 'virial']
            except KeyError:
                pass
            continue
        elif 'config_type' in at.info and at.info['config_type'] == 'dimer':
            # print("got dimer")
            at.info['energy_sigma'] = 0.1
            at.info['force_sigma'] = 0.5
            try:
                del at.info[property_prefix + 'virial']
            except KeyError:
                pass
            continue

        # determine dE energy distance above E,V convex hull
        if 'energy_sigma_override_dE' in at.info:
            dE = at.info["energy_sigma_override_dE"]
            sys.stdout.write("Overriding dE={} in config_type {}\n".format(dE, at.info.get("config_type")))
        else:
            if property_prefix + 'energy' not in at.info:
                continue
            dE = vertical_dist_from_hull(eqns[at.info.get("gap_rss_group")], at.info["fit_sigma_convex_hull_p"])
            # remove so that if config is written, no info field with weird number of components
            # to confused python xyz reader
            del at.info["fit_sigma_convex_hull_p"]
            at.info["fit_sigma_dE"] = dE
            if dE < -0.0001:
                raise ValueError(
                    "Got dE = {} < -0.0001 from convex hull distance for point {} {}, this should never happen".format(
                        dE, at.get_volume() / len(at), at.info[property_prefix + 'energy'] / len(at)))

        # set sigmas from dE
        if dE > 20.0:
            # don't even fit if too high
            for k in property_keys:
                if property_prefix + k in at.info:
                    at.info['ignored_' + property_prefix + k] = at.info[property_prefix + k]
                    del at.info[property_prefix + k]
                if k in at.arrays:
                    at.new_array('ignored_' + property_prefix + k, at.arrays[property_prefix + k])
                    del at.arrays[property_prefix + k]
        else:
            # skip non-periodic configs, volume is meaningless so relation to convex hull is also
            if not all(at.pbc):
                print(
                    "WARNING: modifying sigmas based on distance from convex hull got a non-periodic config that "
                    "wasn't handled specially.  Skipping!")
                continue
            # energy, virial, and Hessian should be scaled by sqrt(N)
            # add factors to keep relationship to previous, incorrect values that ignored this scaling
            root_N_ratio = np.sqrt(len(at)) / np.sqrt(16)
            sigma_E_1 = 0.001
            sigma_E_2 = 0.100
            sigma_set = piecewise_linear(dE,
                                         [(0.1, [sigma_E_1 * root_N_ratio,
                                                 np.sqrt(sigma_E_1),
                                                 2.0 * np.sqrt(sigma_E_1 * root_N_ratio),
                                                 2.0 * np.sqrt(sigma_E_1 * root_N_ratio)]),
                                          (1.0, [sigma_E_2 * root_N_ratio,
                                                 np.sqrt(sigma_E_2),
                                                 2.0 * np.sqrt(sigma_E_2 * root_N_ratio),
                                                 2.0 * np.sqrt(sigma_E_2 * root_N_ratio)])])
            for (field_i, field) in enumerate(['energy_sigma', 'force_sigma', 'virial_sigma', 'hessian_sigma']):
                if "fix_" + field in at.info:
                    sys.stdout.write(
                        "Found fix_{0}, not setting {0}={1}{2}\n".format(
                            field, at.info[field],
                            " in " + at.info["config_type"] if ("config_type" in at.info) else ""))
                else:
                    config_error_scale_factor = at.info.get("fit_error_scale_factor", 1.0)
                    at.info[field] = sigma_set[field_i] * overall_error_scale_factor * config_error_scale_factor * \
                                     field_error_scale_factors.get(field, 1.0)
