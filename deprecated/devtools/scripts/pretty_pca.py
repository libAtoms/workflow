#!/usr/bin/env python3

import json
import os
import re
from argparse import ArgumentParser

import ase.io
import numpy as np
from ase.atoms import Atoms

from wfl.descriptors.quippy import calculate as calc_desc
from wfl.configset import ConfigSet, OutputSpec

parser = ArgumentParser()
desc_grp = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('--imgfile', '-o', help='output image file')
parser.add_argument('--xyz_of_proj_coord', '-x', help='output file with fake coordinates from low-d projection')
desc_grp.add_argument('--desc_str', '-ds', help='descriptor init string')
desc_grp.add_argument('--desc_info_field', '-di', help='info field for descriptor')
desc_grp.add_argument('--desc_array_field', '-da', help='arrays field for descriptor')
parser.add_argument('--n_components', '-n', type=int, help='number of components to reduce to', default=3)
parser.add_argument('--dim_reduce_type', choices=['pca', 'skpca', 'manual_pca'], default='pca')
parser.add_argument('--kernel_zeta', type=float, help='exponent for kernel', default=1)
parser.add_argument('--n_sparse', '-s', type=int, help='number of sparse samples for sparse PCAs', default=-1)
parser.add_argument('--config_type_field', '-t', help='atoms info field for config type, used to color points',
                    default='config_type')
parser.add_argument('--config_type_exclude', '-E', nargs='+', help='config_type group values to exclude', default=[])
parser.add_argument('--type_groups', '-g', help='JSON list pairs of group_name, regexp', default='[]')
parser.add_argument('--reuse', action='store_true', help='reuse calculated descriptors')
parser.add_argument('--subsample', type=int, nargs=2, metavar=('N_min', 'N_subsample'),
                    help='if more than N_min configs in a category substample every N_subsample', default=(-1, 1))
parser.add_argument('files', nargs='+', help='input file[s]')
args = parser.parse_args()

args.config_type_exclude = set(args.config_type_exclude)
args.type_groups = json.loads(args.type_groups)

if args.desc_str is not None:
    if not args.reuse:
        try:
            os.remove('t_w_desc.xyz')
        except:
            pass
    local_desc = not 'average' in args.desc_str
    ats = calc_desc(ConfigSet(input_files=args.files),
                    OutputSpec(output_files='t_w_desc.xyz', force=True, all_or_none=True),
                    descs=args.desc_str, key='CUR_desc', local=local_desc)
    if local_desc:
        args.desc_arrays_field = 'CUR_desc'
    else:
        args.desc_info_field = 'CUR_desc'
else:
    ats = ConfigSet(input_files=args.files)

descs = []
config_types = []
for at in ats:
    conf_type = at.info.get(args.config_type_field, '_NONE_')
    grp = conf_type
    for grp_name, grp_re in args.type_groups:
        if re.search(grp_re, conf_type):
            grp = grp_name
            break
    if grp not in args.config_type_exclude:
        if args.desc_info_field:
            descs.append(at.info[args.desc_info_field])
            config_types.append(grp)
        else:
            descs.extend(at.arrays[args.desc_arrays_field])
            config_types.extend([grp] * len(at))

descs = np.asarray(descs)
print('descs.shape', descs.shape)

#### do dimensionality reduction
if args.dim_reduce_type == 'manual_pca':
    assert args.kernel_zeta == 1

    centers = np.mean(descs, axis=0)
    descs -= centers
    u, s, vh = np.linalg.svd(descs, full_matrices=False)
    # rows of vh are eigenvectors
    proj = descs @ vh[0:args.n_components, :].T

    print(proj.shape)
else:
    reduce_dict = {}
    if args.dim_reduce_type == 'skpca':
        reduce_dict['kpca'] = {'type': 'SPARSE_KPCA',
                               'parameter': {'n_components': args.n_components,
                                             'n_sparse': args.n_sparse,  # -1 == no sparsification
                                             'kernel': {'first_kernel': {'type': 'linear'}}
                                             }
                               }
        if args.kernel_zeta != 1:
            reduce_dict['kpca']['parameter']['kernel']['first_kernel'] = {'type': 'polynomial', 'd': args.kernel_zeta}
    elif args.dim_reduce_type == 'pca':
        reduce_dict['pca'] = {'type': 'PCA', 'parameter': {'n_components': args.n_components, 'scalecenter': False}}
    else:
        raise RuntimeError(f'Unknown dim reduction type {args.dim_reduce_type}')

    from asaplib.reducedim import Dimension_Reducers

    dreducer = Dimension_Reducers(reduce_dict)

    #### actually compute projected low-d coords
    proj = dreducer.fit_transform(descs)

# sort things into groups for plotting
config_types_list = sorted(set(config_types))
config_types_count = []
for conf_type in config_types_list:
    config_types_count.append(sum([t == conf_type for t in config_types]))

config_types_zorder = np.zeros((len(config_types_list)), dtype=int)
config_types_zorder[np.argsort(config_types_count)] = range(len(config_types_list) - 1, -1, -1)

# pick png vs. pdf if not specified
if args.imgfile is None:
    if len(config_types) > 10000:
        args.imgfile = 'low_d_proj.png'
    else:
        args.imgfile = 'low_d_proj.pdf'
# use matplotlib in right mode
import matplotlib

if os.path.splitext(args.imgfile)[1] == '.pdf':
    matplotlib.use('PDF')
from matplotlib import pyplot as plt
import matplotlib.cm

# figure out colormap, normalization
if len(config_types_list) <= 10:
    cmap = 'tab10'
    scatter_kwargs = {'vmin': 0, 'vmax': 10}
elif len(config_types_list) <= 20:
    cmap = 'tab20'
    scatter_kwargs = {'vmin': 0, 'vmax': 20}
else:
    cmap = 'hsv'
    scatter_kwargs = {'vmin': 0, 'vmax': len(config_types_list)}

# do scatterplot, type by type
f = plt.figure()
ax = f.add_subplot()
for type_i, (conf_type, count, zorder) in enumerate(zip(config_types_list, config_types_count, config_types_zorder)):
    print('group', conf_type, count, 'z', zorder)
    if args.subsample[0] > 0 and count > args.subsample[0]:
        subsample_step = args.subsample[1]
    else:
        subsample_step = 1
    d_x = [proj[ii, 0] for ii in range(0, proj.shape[0], subsample_step) if config_types[ii] == conf_type]
    d_y = [proj[ii, 1] for ii in range(0, proj.shape[0], subsample_step) if config_types[ii] == conf_type]
    count = len(d_x)

    s = ax.scatter(d_x, d_y, c=[type_i] * count, s=1, zorder=zorder, cmap=cmap, **scatter_kwargs)
cb = f.colorbar(s, ticks=0.5 + np.arange(len(config_types_list)))
cb.ax.set_yticklabels(config_types_list)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
plt.savefig(args.imgfile)

if args.xyz_of_proj_coord is not None:
    if args.n_components > 3:
        raise RuntimeError('Cannot write fake xyz file for more than 3 components')
    atoms = Atoms()
    for type_i, (conf_type, count, zorder) in enumerate(
            zip(config_types_list, config_types_count, config_types_zorder)):
        if args.subsample[0] > 0 and count > args.subsample[0]:
            subsample_step = args.subsample[1]
        else:
            subsample_step = 1
        data = [proj[ii, :] for ii in range(0, proj.shape[0], subsample_step) if config_types[ii] == conf_type]
        count = len(data)

        pos = np.zeros((count, 3))
        pos[:, 0:len(data[0])] = data
        grp_atoms = Atoms(numbers=[type_i] * count, positions=pos)
        atoms += grp_atoms
    cmap = matplotlib.cm.get_cmap(cmap)
    vtk_commands = []
    for type_i in range(len(config_types_list)):
        (r, g, b, _) = cmap((type_i - scatter_kwargs['vmin']) / (scatter_kwargs['vmax'] - scatter_kwargs['vmin']))
        vtk_commands.append(f'atom_type -r 0.02 -c {r} {g} {b} {type_i}')
    atoms.info['_vtk_commands'] = '; '.join(vtk_commands)
    cell = np.max(atoms.positions, axis=0) - np.min(atoms.positions, axis=0)
    atoms.positions -= (np.max(atoms.positions, axis=0) + np.min(atoms.positions, axis=0)) / 2.0
    cell_scale = min([np.linalg.norm(v) for v in cell if v > 0])
    cell += cell_scale * 0.3
    atoms.positions += cell / 2.0
    atoms.set_cell(cell, False)
    ase.io.write(args.xyz_of_proj_coord, atoms)
