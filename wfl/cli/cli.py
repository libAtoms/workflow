"""
Command line interface to the package
"""

import distutils.util
import glob
import json
import os
import sys
import warnings
from pprint import pformat, pprint

import ase.data
import ase.io
import click
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    import quippy
except ModuleNotFoundError:
    pass
# noinspection PyProtectedMember
from ase.io.extxyz import key_val_str_to_dict
from matplotlib.backends.backend_pdf import PdfPages

from wfl.plotting import reactions_plotting, plot_ef_correlation, plot_2b
from wfl.plotting import normal_modes
from wfl.configset import ConfigSet_in, ConfigSet_out
import wfl.generate_configs.collision
from wfl.generate_configs import vib
import wfl.generate_configs.radicals
import wfl.generate_configs.smiles
import wfl.utils.misc
from wfl.reactions_processing import trajectory_processing
from wfl.select_configs import weighted_cur
import wfl.generate_configs.buildcell
import wfl.select_configs.by_descriptor
import wfl.calc_descriptor

from wfl.utils import gap_xml_tools

from wfl.calculators.dft import evaluate_dft
from wfl.calculators import committee
import wfl.calculators.orca

import wfl.fit.gap_multistage
import wfl.fit.ref_error
import wfl.fit.utils


@click.group("wfl")
@click.option("--verbose", "-v", is_flag=True)
@click.pass_context
def cli(ctx, verbose):
    """GAP workflow command line interface.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # ignore calculator writing warnings
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning, module="ase.io.extxyz")


@cli.group("file-op")
@click.pass_context
def subcli_file_operations(ctx):
    pass


@cli.group("processing")
@click.pass_context
def subcli_processing(ctx):
    pass


@cli.group("plot")
@click.pass_context
def subcli_plot(ctx):
    pass


@cli.group("select-configs")
@click.pass_context
def subcli_select_configs(ctx):
    pass


@cli.group("generate-configs")
@click.pass_context
def subcli_generate_configs(ctx):
    pass


@cli.group("select-configs")
@click.pass_context
def subcli_select_configs(ctx):
    pass


@cli.group("descriptor")
@click.pass_context
def subcli_descriptor(ctx):
    pass


@cli.group("ref-method")
@click.pass_context
def subcli_calculators(ctx):
    pass


@cli.group("fitting")
@click.pass_context
def subcli_fitting(ctx):
    pass



@subcli_generate_configs.command('smiles')
@click.pass_context
@click.argument("smiles", nargs=-1)
@click.option("--output", "-o", help="Output filename, see Configset for details", required=True)
@click.option("--info", "-i", help="Extra info to add to Atoms.info")
@click.option("--force", "-f", help="force writing", is_flag=True)
def configs_from_smiles(ctx, smiles, output, info, force):
    """ ase.Atoms from SMILES string"""

    verbose = ctx.obj["verbose"]

    configset_out = ConfigSet_out(output_files=output, force=force)

    if info is not None:
        info = key_val_str_to_dict(info)

    if verbose:
        print(f'smiles: {smiles}')
        print(f'info: {info}')
        print(configset_out)

    wfl.generate_configs.smiles.run(outputs=configset_out, smiles=smiles, extra_info=info)


@subcli_generate_configs.command('remove-sp3-Hs')
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--outputs", "-o", help="Output filename, see Configset for details", required=True)
@click.option("--force", "-f", help="force writing", is_flag=True)
def configs_from_smiles(ctx, inputs, outputs, force):
    """Removes all sp3 hydrogen atoms for given molecules"""

    verbose = ctx.obj["verbose"]

    inputs = ConfigSet_in(input_files=inputs)
    outputs = ConfigSet_out(output_files=outputs, force=force)

    if verbose:
        print(inputs)
        print(outputs)

    wfl.generate_configs.radicals.abstract_sp3_hydrogen_atoms(inputs=inputs, outputs=outputs)


@subcli_generate_configs.command("collision")
@click.option("--fragments", "--fn", default="fragments.xyz",
              help="Filename of fragments, ase-readable file")
@click.option("--gap-filename", "--gap", default="GAP.xml", help="Filename of GAP xml file")
@click.option("--md-dir", "--dir", default="md", help="Directory for calculation")
@click.option("--velo", default=(0.15, 0.20),
              help="Velocity parameters: (factor, constant). See Supercollider")
@click.option("--nsteps", "-n", default=1000, type=click.INT, help="Number of MD steps")
@click.option("--temp", "-t", default=1000., type=click.FLOAT, help="Temperature in K")
@click.option("--distance", "-d0", "-d", default=6., type=click.FLOAT,
              help="Initial distance of molecules")
@click.option("--collision-kwargs", "--kw", type=click.STRING, required=False,
              help="Kwargs for collision, overwrites options.")
@click.pass_context
def collision(ctx, fragments, gap_filename, md_dir, velo, nsteps, temp, distance, collision_kwargs):
    verbose = ctx.obj["verbose"]
    if verbose:
        print("we are in verbose mode")

    fragment_list = ase.io.read(fragments, ":")
    md_dir = os.path.abspath(md_dir)

    if not os.path.isdir(md_dir):
        if verbose:
            print(f"Creating directory for md: {md_dir}")
        os.mkdir(md_dir)

    collision_kw = dict(velocity_params=velo, nsteps=nsteps, T=temp, d0=distance)
    if collision_kwargs is not None:
        collision_kw.update(key_val_str_to_dict(collision_kwargs))

    wfl.generate_configs.collision.multi_run_all_with_all(
        fragments=fragment_list, param_filename=os.path.abspath(gap_filename), workdir=md_dir,
        **collision_kw)


@subcli_generate_configs.command("neb-ts-irc")
@click.pass_context
@click.argument("seeds", nargs=-1)
@click.option("--gap-filename", "--gap", default="GAP.xml", help="Filename of GAP xml file")
@click.option("--do-neb", is_flag=True, help="Calculate NEB between differing relaxed frames")
@click.option("--do-ts-irc", is_flag=True, help="Calculate TS & IRC on NEBs")
@click.option("--minim-interval", default=50,
              help="Interval of trajectory frames for calculation of relaxations from")
@click.option("--n-pool", "-n", default=None, type=click.INT, help="Number of pooled processes")
@click.option("--minim-kwargs", "--min-kw", type=click.STRING, required=False,
              help="Kwargs for NEB")
@click.option("--neb-kwargs", "--neb-kw", type=click.STRING, required=False, help="Kwargs for NEB")
@click.option("--ts-kwargs", "--ts-kw", type=click.STRING, required=False, help="Kwargs for TS")
@click.option("--irc-kwargs", "--irc-kw", type=click.STRING, required=False, help="Kwargs for IRC")
def trajectory_neb_ts_irc(ctx, seeds, gap_filename, do_neb, do_ts_irc, minim_interval, minim_kwargs,
                          neb_kwargs,
                          ts_kwargs, irc_kwargs, n_pool):
    seeds = [trajectory_processing.xyz_to_seed(fn) for fn in seeds]
    print(seeds)  # debug
    calc = (quippy.potential.Potential, "", dict(param_filename=gap_filename))

    if minim_kwargs is not None:
        minim_kwargs = key_val_str_to_dict(minim_kwargs)
    if neb_kwargs is not None:
        neb_kwargs = key_val_str_to_dict(neb_kwargs)
    if ts_kwargs is not None:
        ts_kwargs = key_val_str_to_dict(ts_kwargs)
    if irc_kwargs is not None:
        irc_kwargs = key_val_str_to_dict(irc_kwargs)

    wfl.generate_configs.collision.post_process_collision(
        seed=seeds, calc=calc, do_neb=do_neb, do_ts_irc=do_ts_irc, minim_kwargs=minim_kwargs,
        minim_interval=minim_interval, neb_kwargs=neb_kwargs,
        ts_kwargs=ts_kwargs, irc_kwargs=irc_kwargs, n_pool=n_pool,
    )


@subcli_file_operations.command("gather")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output", "-o", help="Output filename, see Configset for details", required=True)
@click.option("--index", "-i", type=click.STRING, required=False,
              help="Pass this index to configset globally")
@click.option("--force", "-f", help="force writing", is_flag=True)
def file_gather(ctx, inputs, output, force, index):
    """ Gathers configurations from files through a Configset
    """
    verbose = ctx.obj["verbose"]

    # ignore calculator writing warnings
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning, module="ase.io.extxyz")

    configset_in = ConfigSet_in(input_files=inputs, default_index=index)
    configset_out = ConfigSet_out(output_files=output, force=force)

    if verbose:
        print(configset_in)
        print(configset_out)

    configset_out.write(configset_in)
    configset_out.end_write()


@subcli_file_operations.command("strip")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--keep-info", "-i", required=False, multiple=True,
              help="Keys to keep from info if present")
@click.option("--keep-array", "-a", required=False, multiple=True,
              help="Keys to keep from arrays if present")
@click.option("--cell", is_flag=True, help="Keep the cell w/ PBC")
@click.option("--output", "-o", help="Output filename, see Configset for details",
              type=click.STRING, required=False)
@click.option("--force", "-f", help="force writing", is_flag=True)
def strip(ctx, inputs, keep_info, keep_array, cell, output, force):
    """Strips structures of all info and arrays except the ones specified to keep

    Notes
    -----
    can be replaced by a call on `ase convert` when that allows for taking None of the
    info/arrays keys
    see issue: https://gitlab.com/ase/ase/-/issues/727
    """
    verbose = ctx.obj["verbose"]

    if output is None:
        if not force:
            raise ValueError(
                "Error in: `wfl file-op strip`: neither output nor force are given, specify one "
                "at least")
        output = inputs

    configset_in = ConfigSet_in(input_files=inputs)
    configset_out = ConfigSet_out(output_files=output, force=force, all_or_none=True)

    # iterate, used for both progressbar and without the same way
    for at in configset_in:
        new_at = ase.Atoms(at.get_chemical_symbols(), positions=at.get_positions())

        if cell:
            new_at.set_cell(at.get_cell())
            new_at.set_pbc(at.get_pbc())

        if keep_info is not None:
            for key, val in at.info.items():
                if key in keep_info:
                    new_at.info[key] = val

        if keep_array is not None:
            for key, val in at.arrays.items():
                if key in keep_array:
                    new_at.arrays[key] = val

        configset_out.write(new_at, configset_in.get_current_input_file())

    configset_out.end_write()


@subcli_processing.command("committee")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--prefix", "-p", help="Prefix to put on the first", type=click.STRING,
              default="committee.")
@click.option("--gap-fn", "-g", type=click.STRING, help="gap filenames, globbed", required=True,
              multiple=True)
@click.option("--stride", "-s", "-n", help="Take every Nth frame only", type=click.STRING,
              required=False)
@click.option("--force", "-f", help="force writing", is_flag=True)
def calc_ef_committee(ctx, inputs, prefix, gap_fn, stride, force):
    """Calculated energy and force with a committee of models.
    Uses a prefix on the filenames, this can be later changed
    does not support globbed filenames as xyz input

    configset works in many -> many mode here
    """

    verbose = ctx.obj["verbose"]

    # apply the prefix to output names
    outputs = {fn: os.path.join(os.path.dirname(fn), f"{prefix}{os.path.basename(fn)}") for fn in
               inputs}

    configset_in = ConfigSet_in(input_files=inputs, default_index=stride)
    configset_out = ConfigSet_out(output_files=outputs, force=force)

    if verbose:
        print(configset_in)
        print(configset_out)

    # read GAP models
    gap_fn_list = []
    for fn in gap_fn:
        gap_fn_list.extend(sorted(glob.glob(fn)))
    gap_model_list = [(quippy.potential.Potential, "", dict(param_filename=fn)) for fn in gap_fn_list]

    # calculate E,F
    for at in configset_in:
        at = committee.calculate_committee(at, gap_model_list)
        configset_out.write(at, configset_in.get_current_input_file())

    configset_out.end_write()


@subcli_processing.command("max-similarity")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--train-filename", "--tr", help="Training filename - reference to use",
              type=click.STRING,
              required=False)
@click.option("--cutoff-list", "-c", help="Cutoffs to calculate SOAP with", type=click.STRING,
              multiple=True,
              default=("3.0", "6.0"), required=False)
@click.option("--z-list", "-z",
              help="Atomic numbers to include in SOAP - incomplete implementation atm.",
              type=click.INT, multiple=True, default=(1, 6, 8), required=False)
@click.option("--prefix", "-p", help="Prefix to put on the first", type=click.STRING,
              default="sim.")
@click.option("--force", "-f", help="force writing", is_flag=True)
def calc_max_kernel_similarity(ctx, inputs, force, train_filename, cutoff_list, z_list, prefix):
    """Calculate maximum kernel similarity with a given training set.

    configset is working in many -> many mode
    """
    verbose = ctx.obj["verbose"]

    # apply the prefix to output names
    outputs = {fn: os.path.join(os.path.dirname(fn), f"{prefix}{os.path.basename(fn)}") for fn in
               inputs}

    configset_in = ConfigSet_in(input_files=inputs)
    configset_out = ConfigSet_out(output_files=outputs, force=force)
    if verbose:
        print(configset_in)
        print(configset_out)
        sys.stdout.flush()

    # initialisations for calculation
    frames_train = ase.io.read(train_filename, ":")
    soap_dict, desc_ref = trajectory_processing.create_desc_ref(frames_train, cutoff_list, z_list)
    if verbose:
        print("Calculated SOAP vectors for training set")

    for at in configset_in:
        at = trajectory_processing.calc_max_similarity_atoms(at, soap_dict, desc_ref)
        configset_out.write(at, from_input_file=configset_in.get_current_input_file())

    configset_out.end_write()


@subcli_select_configs.command("weighted-cur")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--cut-threshold", type=click.FLOAT, default=None,
              help="Cut each trajectory when combined similarity metric [exp(mean(log(k), "
                   "over atoms)] falls "
                   "below threshold.")
@click.option("--limit", "-l", type=click.FLOAT, required=True,
              help="Energy limit below which lower weight is given, above it it is linear")
@click.option("--output", "-o", required=True, help="Output filename, see Configset for details", )
@click.option("--stride", "-i", type=click.INT, required=False,
              help="Pass this index to configset globally")
@click.option("--descriptor", "-d", type=click.STRING, multiple=True,
              default=(
              "soap n_max=8 l_max=6 atom_sigma=0.3 n_species=3 species_Z={6 1 8 } cutoff=3.0",),
              help="SOAP string, species-Z will be added to this!")
@click.option("--n-select", "-n", type=(click.INT, click.INT), multiple=True,
              help="(z, num) tuples of how many to select per atomic number, per cutoff instance")
@click.option("--force", "-f", help="force writing", is_flag=True)
def select_cur_and_committee(ctx, inputs, output, cut_threshold, limit, descriptor, n_select, force,
                             stride):
    """Selection with weighting CUR with energy std of committee of models after cutting with
    global metric on traj

    Notes
    -----
    - This is too specific now, should maybe split the trajectory manipulation and CUR into two
    and merge tha latter
    into the general CUR we have here
    """
    verbose = ctx.obj["verbose"]

    configset_in = ConfigSet_in(input_files=inputs,
                                default_index=(f"::{stride}" if stride is not None else ":"))
    if cut_threshold is not None:
        # cutting by global SOAP metric -- simply recreating the configset with indices calculated
        new_inputs = []

        for subcfs in configset_in.group_iter():
            idx = trajectory_processing.cut_trajectory_with_global_metric(subcfs, cut_threshold)
            current_fn = configset_in.get_current_input_file()
            if verbose:
                print(f"cutting at index: {idx} on file {current_fn}")

            if stride is None:
                if idx == -1:
                    str_index = ":"
                else:
                    str_index = f":{idx}"
            else:
                if idx == -1:
                    str_index = f"::{stride}"
                else:
                    str_index = f":{idx * stride}:{stride}"

            new_inputs.append((current_fn, str_index))

        # recreate the configset to have the file indices in it
        configset_in = ConfigSet_in(input_files=new_inputs)

    configset_out = ConfigSet_out(output_files=output, force=force)

    if verbose:
        print(configset_in)
        print(configset_out)
        sys.stdout.flush()

    z_list = []
    num_dict = dict()
    for z, num in n_select:
        z_list.append(z)
        num_dict[z] = num
    if verbose:
        print("(z, num) to take:", num_dict)

    weighted_cur.selection(configset_in, configset_out, z_list, descriptor, limit, num_dict)
    configset_out.end_write()



@subcli_plot.command("ef-thinpoints")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--plot-fn", type=click.STRING, default="plot_thinpoints.pdf",
              help="Filename for plot")
@click.option("--gap-fn", type=click.STRING, default="GAP.xml", help="Filename of GAP, used for e0")
@click.option("--ref-energy-key", "-re", type=click.STRING, default="gap_energy",
              help="Ref energy key")
@click.option("--gap-energy-key", "-ge", type=click.STRING, default="ref_energy",
              help="GAP energy key")
@click.option("--ref-force-key", "-rf", type=click.STRING, default="ref_forces",
              help="Ref force key")
@click.option("--gap-force-key", "-gf", type=click.STRING, default="gap_forces",
              help="GAP force key")
@click.option("--colors", type=click.STRING, help="Colors dict as str")
@click.option("--thin-kwargs", type=click.STRING, help="Kwargs for thinpoints")
def plot_ef_thinpoints(ctx, inputs, plot_fn, gap_fn, colors, thin_kwargs, ref_energy_key,
                       gap_energy_key,
                       ref_force_key, gap_force_key):
    """Plot energy and force correlation with thinpoints
    """
    frames = ConfigSet_in(input_files=inputs)
    print(frames)
    e0 = gap_xml_tools.extract_e0(gap_fn)
    if colors is not None:
        colors = key_val_str_to_dict(colors)
    if thin_kwargs is not None:
        thin_kwargs = key_val_str_to_dict(thin_kwargs)

    plot_ef_correlation.plot(frames=frames, fig_filename=plot_fn, e0=e0,
                             ref_energy_key=ref_energy_key, ref_force_key=ref_force_key,
                             gap_energy_key=gap_energy_key, gap_force_key=gap_force_key,
                             colors=colors, thin_params=thin_kwargs)


@subcli_plot.command("2b")
@click.pass_context
@click.argument("atomic-numbers", nargs=-1, type=click.INT)
@click.option("--plot-fn", type=click.STRING, default="plot_2b.pdf", help="Filename for plot")
@click.option("--gap-fn", type=click.STRING, default="GAP.xml", help="Filename of GAP")
@click.option("--baseline-fn", type=click.STRING, help="Filename of baseline model")
@click.option("--baseline-label", type=click.STRING, default="glue",
              help="label of baseline model for plot")
@click.option("--baseline-kw", type=click.STRING, help="kwargs for initialising baseline model")
@click.option("--colors", type=click.STRING, help="Colors dict as str")
@click.option("--cutoff", type=click.FLOAT, help="Cutoff un to which to calculate binding energies")
def plot_2b_binding(ctx, atomic_numbers, plot_fn, gap_fn, colors, baseline_fn, baseline_label,
                    baseline_kw,
                    cutoff=None):
    if colors is not None:
        colors = key_val_str_to_dict(colors)

    gap = (quippy.potential.Potential, "", dict(param_filename=gap_fn))
    e0 = gap_xml_tools.extract_e0(gap_fn)

    # only create baseline if filename is given
    if baseline_fn is not None:
        if baseline_kw is None:
            baseline_kw = dict()
            if baseline_label == "glue":
                baseline_kw["args_str"] = "IP Glue"
        else:
            baseline_kw = key_val_str_to_dict(baseline_kw)
        baseline_calc = quippy.potential.Potential(param_filename=baseline_fn, **baseline_kw)
    else:
        baseline_calc = None

    plot_2b.plot_2b_multipage(gap_calculator=gap, baseline_calculator=baseline_calc,
                              fig_filename=plot_fn, e0=e0,
                              atomic_numbers=atomic_numbers, cutoff=cutoff,
                              baseline_label=baseline_label,
                              colors=colors)


@subcli_plot.command("traj-std")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--plot-fn", type=click.STRING, default="plot.pdf",
              help="Filename for plot, multi-page")
@click.option("--data-fn", type=click.STRING, default="data.npy",
              help="Filename for compressed data storage")
@click.option("--enforce-rebuild", "-f", is_flag=True,
              help="Enforce rebuilding of the compressed data")
def plot_trajectory_std(ctx, inputs, data_fn, enforce_rebuild, plot_fn):
    """Plots the estimated STD of energy and force predictions with a committee of models

    Uses a compressed data file to keep the extracted information about each trajectory for later
    """

    verbose = ctx.obj["verbose"]

    inputs = sorted(inputs)
    if verbose:
        print(f"input files, count: {len(inputs)}")
        pprint(inputs)

    if not enforce_rebuild and os.path.isfile(data_fn):
        # read back
        data = np.atleast_1d(np.load(data_fn, allow_pickle=True))[0]
    else:
        data = trajectory_processing.extract_data_from_trajectories(inputs)
        np.save(data_fn, data)
        print("read and saved the data")

    with PdfPages(plot_fn) as pdf:
        for j, chunk in enumerate(wfl.utils.misc.chunks(inputs, 8)):
            if verbose:
                print("plot{:4}".format(j))
            page_fig = reactions_plotting.plot_one_page_figure(data, chunk)
            pdf.savefig(page_fig)
            plt.close(page_fig)


@subcli_plot.command("traj-kernel")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--plot-fn", type=click.STRING, default="plot_kernel.pdf",
              help="Filename for plot, multi-page")
def plot_trajectory_std(ctx, inputs, plot_fn):
    """Plots the kernel similarity of a trajectory with the first frame
    """

    verbose = ctx.obj["verbose"]

    inputs = sorted(inputs)
    if verbose:
        print(f"input files, count: {len(inputs)}")
        pprint(inputs)

    with PdfPages(plot_fn) as pdf:
        for j, filename in enumerate(inputs):
            if verbose:
                print("plot{:4}".format(j))
            page_fig = reactions_plotting.calc_and_plot_kernel_vs_first(filename, filename,
                                                                        index="::8")
            pdf.savefig(page_fig)
            plt.close(page_fig)

@subcli_plot.command('normal-modes')
@click.pass_context
@click.option('--file-1', '-f1', required=True, type=click.Path(exists=True),
              help = 'xyz with normal mode information')
@click.option('--prefix-1', '-p1', required=True, help='prefix for normal modes properties in xyz file')
@click.option('--label-1', '-l1', help='axis label')
@click.option('--file-2', '-f2', required=True, type=click.Path(exists=True),
              help = 'xyz with normal mode information')
@click.option('--prefix-2', '-p2', required=True, help='prefix for normal modes properties in xyz file')
@click.option('--label-2', '-l2', help='axis label')
@click.option('--plot-fn', default='eigenvector_plot.png', show_default=True,
              help='figure filename')
@click.option('--arrange-by', type=click.Choice(['frequency', 'order']),
              help='heatmap with eigenvectors arranged in order or a scatter plot of '
                   'frequency vs frequency', default='frequency',
              show_default=True)
@click.option('--cmap', default='darkred', show_default=True,
              help='matplotlib color or colormap name')
@click.option('--adjust-color-range', is_flag=True,
              help='pick colormap range based on dot product values')
@click.option('--tolerance', default=1e-4, type=click.FLOAT,
              show_default=True,
              help='allowance for dot products to fall outside [0, 1] range')
def compare_normal_modes(ctx, file_1, prefix_1, file_2, prefix_2, label_1, label_2,
                         plot_fn, arrange_by, cmap, adjust_color_range, tolerance):
    """Plot eigenvector dot products and frequencies"""

    normal_modes.eigenvector_plot(nm1=file_1, nm2=file_2, prop_prefix1=prefix_1,
                                  prop_prefix2=prefix_2, label1=label_1,
                                  label2=label_2, fig_fname=plot_fn,
                                  arrange_by=arrange_by, cmap=cmap,
                                  adjust_color_range=adjust_color_range,
                                  tolerance=tolerance)

@subcli_generate_configs.command("repeat-buildcell")
@click.pass_context
@click.option("--output-file", type=click.STRING)
@click.option("--output-all-or-none", is_flag=True)
@click.option("--buildcell-input", type=click.STRING, required=True, help="buildcell input file")
@click.option("--buildcell-exec", type=click.STRING, required=True,
              help="buildcell executable including path")
@click.option("--n-configs", "-N", type=click.INT, required=True,
              help="number of configs to generate")
@click.option("--extra-info", type=click.STRING, default="",
              help="dict of information to store in Atoms.info")
@click.option("--perturbation", type=click.FLOAT, default=0.0,
              help="magnitude of random perturbation to atomic positions")
def _repeat_buildcell(ctx, output_file, output_all_or_none, buildcell_input, buildcell_exec,
                      n_configs,
                      extra_info, perturbation):
    """Repeatedly runs buildcell (from Pickard's AIRSS distribution) to generate random configs with
    specified species, volumes, distances, symmetries, etc.

    Minimal contents of --buildcell-input file:

    \b
    #TARGVOL=<min_vol>-<max_vol> (NOTE: volume is volume_per_formula_unit/number_of_species)
    #SPECIES=<elem_symbol_1>%NUM=<num_1>[,<elem_symbol_2>%NUM=<num_2 ...]
    #NFORM=[ <n_min>-<n_max> | { <n_1>, <n_2>, ... } ]
    #SYMMOPS=<n_min>-<n_max> (NOTE: optional)
    #SLACK=0.25
    #OVERLAP=0.1
    #COMPACT
    #MINSEP=<min_separation_default> <elem_symbol_1>-<elem_symbol_1>=<min_separation_1_1> [
    <elem_symbol_1>-<elem_symbol_2=<min_separation_1_2> ... ]
    ##EXTRA_INFO <info_field>=<value> (NOTE: optional)
    """
    extra_info = key_val_str_to_dict(extra_info)
    with open(buildcell_input) as bc_f:
        buildcell_input_txt = bc_f.read()

    wfl.generate_configs.buildcell.run(
        outputs=ConfigSet_out(output_files=output_file, all_or_none=output_all_or_none),
        config_is=range(n_configs),
        buildcell_cmd=buildcell_exec,
        buildcell_input=buildcell_input_txt,
        extra_info=extra_info,
        perturbation=perturbation,
        verbose=ctx.obj["verbose"]
    )


@subcli_select_configs.command("CUR-global")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--output-all-or-none", is_flag=True)
@click.option("--n-configs", "-N", type=click.INT, required=True,
              help="number of configs to select")
@click.option("--descriptor-key", type=click.STRING, required=True,
              help="Atoms.info key for descriptor vector")
@click.option("--kernel_exponent", type=click.FLOAT, help="exponent of dot-product for kernel")
@click.option("--deterministic", is_flag=True,
              help="use deterministic (not stochastic) CUR selection")
def _CUR_global(ctx, inputs, output_file, output_all_or_none, descriptor_key, kernel_exponent,
                n_configs,
                deterministic):
    wfl.select_configs.by_descriptor.CUR_conf_global(
        inputs=ConfigSet_in(input_files=inputs),
        outputs=ConfigSet_out(output_files=output_file, all_or_none=output_all_or_none),
        num=n_configs,
        at_descs_info_key=descriptor_key, kernel_exp=kernel_exponent, stochastic=not deterministic)


@subcli_descriptor.command("calc")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--output-all-or-none", is_flag=True)
@click.option("--descriptor", type=click.STRING, required=True, help="quippy.Descriptor arg string")
@click.option("--key", type=click.STRING, required=True,
              help="key to store in Atoms.info (global) or Atoms.arrays(local)")
@click.option("--local", is_flag=True, help="calculate a local (per-atom) descriptor")
@click.option("--force", is_flag=True, help="overwrite existing info or arrays item if present")
def _calc_descriptor(ctx, inputs, output_file, output_all_or_none, descriptor, key, local, force):
    wfl.calc_descriptor.calc(
        inputs=ConfigSet_in(input_files=inputs),
        outputs=ConfigSet_out(output_files=output_file, all_or_none=output_all_or_none),
        descs=descriptor,
        key=key,
        local=local,
        force=force
    )


@subcli_calculators.command("vasp-eval")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--output-all-or-none", is_flag=True)
@click.option("--base-rundir", type=click.STRING, help='directory to run jobs in')
@click.option("--directory-prefix", type=click.STRING, default='run_VASP_')
@click.option("--output-prefix", type=click.STRING)
@click.option("--properties", type=click.STRING, default='energy,forces,stress')
@click.option("--incar", type=click.STRING, help='INCAR file, optional')
@click.option("--kpoints", type=click.STRING, help='KPOINTS file, optional')
@click.option("--vasp-kwargs", type=click.STRING, default="isym=0 isif=7 nelm=300 ediff=1.0e-7",
              help='QUIP-style key-value pairs for ASE vasp calculator kwargs that override contents of INCAR and KPOINTS if both are provided.'
                   '"pp", which is normallly XC-based dir to put between VASP_PP_PATH and POTCAR dirs defaults to ".". Key VASP_PP_PATH will be '
                   'used to set corresponding env var, which is used as dir above <chem_symbol>/POTCAR')
@click.option("--vasp-command", type=click.STRING)
def _vasp_eval(ctx, inputs, output_file, output_all_or_none, base_rundir, directory_prefix,
               output_prefix, properties,
               incar, kpoints, vasp_kwargs, vasp_command):
    vasp_kwargs = key_val_str_to_dict(vasp_kwargs)
    vasp_kwargs['INCAR_file'] = incar
    vasp_kwargs['KPOINTS_file'] = kpoints
    evaluate_dft(
        inputs=ConfigSet_in(input_files=inputs),
        outputs=ConfigSet_out(output_files=output_file, all_or_none=output_all_or_none),
        calculator_name="VASP",
        base_rundir=base_rundir,
        dir_prefix=directory_prefix,
        output_prefix=output_prefix,
        properties=properties.split(','),
        calculator_kwargs=vasp_kwargs,
        calculator_command=vasp_command)



@subcli_calculators.command("aims-eval")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--output-all-or-none", is_flag=True)
@click.option("--output-prefix", type=click.STRING, default="", help="prefix in info/arrays for results")
@click.option("--base-rundir", type=click.STRING, help="directory to put all calculation directories into")
@click.option("--directory-prefix", type=click.STRING, default='run_AIMs_')
@click.option("--properties", type=click.STRING, default='energy',
              help="properties to calculate, string is split")
@click.option("--aims-command", type=click.STRING, help="command, including mpirun")
@click.option("--aims-kwargs-file", type=click.STRING, help="AIMs keywords, passed as dict")
@click.option("--keep-files", type=click.STRING, default="default",
              help="How much of files to keep, default is NOMAD compatible subset")
@click.option("--force", is_flag=True)
def _aims_eval(ctx, inputs, output_file, output_all_or_none, base_rundir, directory_prefix, properties,
                 aims_command, aims_kwargs_file, keep_files, output_prefix, force):

    with open(aims_kwargs_file) as fp:
        aims_kwargs = json.load(fp)

    if len(output_prefix)==0:
        output_prefix=None

    evaluate_dft(
        inputs=ConfigSet_in(input_files=inputs),
        outputs=ConfigSet_out(output_files=output_file, all_or_none=output_all_or_none, force=force),
        calculator_name="AIMS",
        base_rundir=base_rundir,
        dir_prefix=directory_prefix,
        properties=properties.split(),
        calculator_command=aims_command,
        keep_files=keep_files,
        calculator_kwargs=aims_kwargs,
        output_prefix=output_prefix
    )


    
@subcli_calculators.command("castep-eval")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--output-all-or-none", is_flag=True)
@click.option("--output-prefix", type=click.STRING, help="prefix in info/arrays for results")
@click.option("--base-rundir", type=click.STRING, help="directory to put all calculation directories into")
@click.option("--directory-prefix", type=click.STRING, default='run_CASTEP_')
@click.option("--properties", type=click.STRING, default='energy forces stress',
              help="properties to calculate, string is split")
@click.option("--castep-command", type=click.STRING, help="command, including appropriate mpirun")
@click.option("--castep-kwargs", type=click.STRING, help="CASTEP keywords, passed as dict")
@click.option("--keep-files", type=click.STRING, default="default",
              help="How much of files to keep, default is NOMAD compatible subset")
def _castep_eval(ctx, inputs, output_file, output_all_or_none, base_rundir, directory_prefix, properties,
                 castep_command, castep_kwargs, keep_files, output_prefix):
    if castep_kwargs is not None:
        castep_kwargs = key_val_str_to_dict(castep_kwargs)

    evaluate_dft(
        inputs=ConfigSet_in(input_files=inputs),
        outputs=ConfigSet_out(output_files=output_file, all_or_none=output_all_or_none),
        calculator_name="CASTEP",
        base_rundir=base_rundir,
        dir_prefix=directory_prefix,
        properties=properties.split(),
        calculator_command=castep_command,
        keep_files=keep_files,
        calculator_kwargs=castep_kwargs,
        output_prefix=output_prefix
    )


@subcli_calculators.command("orca-eval-basin-hopping")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--output-all-or-none", is_flag=True)
@click.option("--output-prefix", type=click.STRING, help="prefix in info/arrays for results")
@click.option("--base-rundir", type=click.STRING,
              help="directory to put all calculation directories into")
@click.option("--directory-prefix", type=click.STRING, default='ORCA')
@click.option("--calc-kwargs", "--kw", type=click.STRING, required=False,
              default=None,
              help="Kwargs for calculation, overwritten by other options")
@click.option("--keep-files", type=click.STRING, default="default",
              help="How much of files to keep, default is NOMAD compatible subset")
@click.option("--orca-command", type=click.STRING, help="path to ORCA executable, default=`orca`")
@click.option("--scratch-path", "-tmp", type=click.STRING,
              help="Directory to use as scratch for calculations, SSD recommended, default: cwd")
@click.option("--n-run", "-nr", type=click.INT, required=True,
              help="Number of global optimisation runs for each frame")
@click.option("--n-hop", "-nh", type=click.INT, required=True,
              help="Number of hopping steps to take per run")
@click.option("--orca-simple-input", type=click.STRING,
              help="orca simple input line, make sure it is correct, default "
                   "is recPBE with settings tested for radicals")
@click.option("--orca-additional-blocks", type=click.STRING,
              help="orca blocks to be added, default is None")
def orca_eval(ctx, inputs, base_rundir, output_file, output_all_or_none, directory_prefix,
              orca_command, calc_kwargs, keep_files, output_prefix, scratch_path, n_run, n_hop,
              orca_simple_input, orca_additional_blocks):
    verbose = ctx.obj["verbose"]

    if scratch_path is not None:
        if not os.path.isdir(scratch_path):
            raise NotADirectoryError(
                f"Scratch path needs to be a directory, invalid given: {scratch_path}")
        if not os.access(scratch_path, os.W_OK):
            raise PermissionError(f"cannot write to specified scratch dir: {scratch_path}")
        scratch_path = os.path.abspath(scratch_path)

    try:
        keep_files = bool(distutils.util.strtobool(keep_files))
    except ValueError:
        if keep_files != 'default':
            raise RuntimeError(f'invalid value given for "keep_files" ({keep_files})')

    # default: dict()
    if calc_kwargs is None:
        calc_kwargs = dict()
    else:
        calc_kwargs = key_val_str_to_dict(calc_kwargs)

    # update args
    for key, val in dict(orca_command=orca_command, scratch_path=scratch_path, n_run=n_run,
                         n_hop=n_hop, orcasimpleinput=orca_simple_input,
                         orcablock=orca_additional_blocks).items():
        if val is not None:
            calc_kwargs[key] = val

    configset_in = ConfigSet_in(input_files=inputs)
    configset_out = ConfigSet_out(output_files=output_file, all_or_none=output_all_or_none)

    if verbose:
        print(configset_in)
        print(configset_out)
        print("ORCA wfn-basin hopping calculation parameters: ", calc_kwargs)

    wfl.calculators.orca.evaluate_basin_hopping(
        inputs=configset_in, outputs=configset_out, base_rundir=base_rundir, dir_prefix=directory_prefix,
        keep_files=keep_files, output_prefix=output_prefix, orca_kwargs=calc_kwargs
    )


@subcli_calculators.command("orca-eval")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--output-file", type=click.STRING, required=True)
@click.option("--output-all-or-none", is_flag=True)
@click.option("--output-prefix", type=click.STRING, help="prefix in info/arrays for results")
@click.option("--base-rundir", type=click.STRING, help="directory to put all calculation directories into")
@click.option("--directory-prefix", type=click.STRING, default='ORCA')
@click.option("--calc-kwargs", "--kw", type=click.STRING, required=False, default=None,
              help="Kwargs for calculation, overwritten by other options")
@click.option("--keep-files", type=click.STRING, default="default",
              help="How much of files to keep, default is NOMAD compatible subset")
@click.option("--orca-command", type=click.STRING, help="path to ORCA executable, default=`orca`")
@click.option("--scratch-path", "-tmp", type=click.STRING,
              help="Directory to use as scratch for calculations, SSD recommended, default: cwd")
@click.option("--orca-simple-input", type=click.STRING, help="orca simple input line, make sure it is correct, default "
                                                             "is recPBE with settings tested for radicals")
@click.option("--orca-additional-blocks", type=click.STRING, help="orca blocks to be added, default is None")
def orca_eval(ctx, inputs, base_rundir, output_file, output_all_or_none, directory_prefix,
              orca_command, calc_kwargs, keep_files, output_prefix, scratch_path,
              orca_simple_input, orca_additional_blocks):
    verbose = ctx.obj["verbose"]

    if scratch_path is not None:
        if not os.path.isdir(scratch_path):
            raise NotADirectoryError(f"Scratch path needs to be a directory, invalid given: {scratch_path}")
        if not os.access(scratch_path, os.W_OK):
            raise PermissionError(f"cannot write to specified scratch dir: {scratch_path}")
        scratch_path = os.path.abspath(scratch_path)

    try:
        keep_files = bool(distutils.util.strtobool(keep_files))
    except ValueError:
        if keep_files != 'default':
            raise RuntimeError(f'invalid value given for "keep_files" ({keep_files})')

    # default: dict()
    if calc_kwargs is None:
        calc_kwargs = dict()
    else:
        calc_kwargs = key_val_str_to_dict(calc_kwargs)

    # update args
    for key, val in dict(orca_command=orca_command, scratch_path=scratch_path,
                         orcasimpleinput=orca_simple_input, orcablocks=orca_additional_blocks).items():
        if val is not None:
            calc_kwargs[key] = val

    configset_in = ConfigSet_in(input_files=inputs)
    configset_out = ConfigSet_out(output_files=output_file, all_or_none=output_all_or_none)

    if verbose:
        print(configset_in)
        print(configset_out)
        print("ORCA calculation parameters: ", calc_kwargs)

    wfl.calculators.orca.evaluate(
        inputs=configset_in, outputs=configset_out, base_rundir=base_rundir,
        dir_prefix=directory_prefix,
        keep_files=keep_files, output_prefix=output_prefix, orca_kwargs=calc_kwargs
    )


@subcli_processing.command("reference-error")
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--pre-calc", "-pre", type=click.STRING, required=True,
              help="string to exec() that sets up for calculator, e.g. imports")
@click.option("--calc", "-c", type=click.STRING, required=True,
              help="string to 'eval()' that returns calculator constructor")
@click.option("--calc-args", "-a", type=click.STRING,
              help="json list of calculator constructor args", default="[]")
@click.option("--calc-kwargs", "-k", type=click.STRING,
              help="json dict of calculator constructor kwargs", default="{}")
@click.option("--ref-prefix", "-r", type=click.STRING,
              help="string to prepend to info/array keys for reference energy, "
                   "forces, virial. If None, use SinglePointCalculator results")
@click.option("--properties", "-p", type=click.STRING,
              help="command separated list of properties to use",
              default='energy_per_atom,forces,virial_per_atom')
@click.option("--category_keys", "-C", type=click.STRING,
              help="comma separated list of info keys to use for doing per-category error",
              default="")
@click.option("--outfile", "-o", type=click.STRING, help="output file, - for stdout", default='-')
@click.option("--intermed-file", type=click.STRING,
              help="intermediate file to contain calculator results, keep in memory if None")
def ref_error(ctx, inputs, pre_calc, calc, calc_args, calc_kwargs, ref_prefix, properties,
              category_keys, outfile, intermed_file):
    verbose = ctx.obj["verbose"]

    cs_out = ConfigSet_out(output_files=intermed_file)

    if pre_calc is not None:
        exec(pre_calc)

    if ref_prefix is None:
        # copy from SinglePointCalculator to info/arrays so calculator results won't overwrite
        # will do this by keeping copy of configs in memory, maybe should have an optional way to do
        # this via a file instead.
        inputs = list(ConfigSet_in(input_files=inputs))
        ref_property_keys = wfl.fit.utils.copy_properties(inputs, ref_property_keys=ref_prefix)
        inputs = ConfigSet_in(input_configs=inputs)
    else:
        ref_property_keys = {p: ref_prefix + p for p in
                             ['energy', 'forces', 'stress', 'virial']}
        inputs = ConfigSet_in(input_files=inputs)

    errs = wfl.fit.ref_error.calc(inputs, cs_out,
                                  calculator=(
                                  eval(calc), json.loads(calc_args), json.loads(calc_kwargs)),
                                  ref_property_keys=ref_property_keys,
                                  properties=[p.strip() for p in properties.split(',')],
                                  category_keys=category_keys.split(', '))

    if outfile == '-':
        pprint(errs)
    else:
        with open(outfile, 'w') as fout:
            fout.write(pformat(errs) + '\n')


@subcli_fitting.command("multistage-gap")
@click.pass_context
@click.argument("inputs", nargs=-1, required=True)
@click.option("--GAP-name", "-G", type=click.STRING, required=True,
              help="name of final GAP file, not including xml suffix")
@click.option("--params-file", "-P", type=click.STRING, required=True,
              help="fit parameters JSON file")
@click.option("--property_prefix", "-p", type=click.STRING,
              help="prefix to reference property keys")
@click.option("--database-modify-mod", "-m", type=click.STRING,
              help="python module that defines a 'modify()' function for operations like setting "
                   "per-config fitting error")
@click.option("--run-dir", "-d", type=click.STRING, help="subdirectory to run in")
@click.option("--fitting-error/--no-fitting-error", help="calculate error for fitting configs",
              default=True)
@click.option("--testing-configs", type=click.STRING,
              help="space separated list of files with testing configurations to calculate error for")
def multistage_gap(ctx, inputs, gap_name, params_file, property_prefix, database_modify_mod,
                   run_dir, fitting_error,
                   testing_configs):
    verbose = ctx.obj["verbose"]

    with open(params_file) as fin:
        fit_params = json.load(fin)

    if testing_configs is not None:
        testing_configs = ConfigSet_in(input_files=testing_configs.split())

    GAP, fit_err, test_err = wfl.fit.gap_multistage.fit(ConfigSet_in(input_files=inputs),
                                                        GAP_name=gap_name, params=fit_params,
                                                        ref_property_prefix=property_prefix,
                                                        database_modify_mod=database_modify_mod,
                                                        calc_fitting_error=fitting_error,
                                                        testing_configs=testing_configs,
                                                        run_dir=run_dir, verbose=verbose)


@subcli_fitting.command('simple-gap')
@click.pass_context
@click.option('--gap-file', '-g', default='GAP.xml', show_default=True,
              help='GAP filename, overrides option'
                   'in parameters file')
@click.option('--atoms-filename',
              help='xyz with training configs and isolated atoms')
@click.option('--param-file', '-p', required=True,
              help='yml file with gap parameters ')
@click.option('--gap-fit-exec', default='gap_fit',
              help='executable for gap_fit')
@click.option('--output-file', '-o', default='default',
              help='filename where to save gap output, defaults to '
                   'gap_basename + _output.txt')
@click.option('--fit', is_flag=True, help='Actually run the gap_fit command')
@click.option("--verbose", "-v", is_flag=True)
def simple_gap_fit(ctx, gap_file, atoms_filename, param_file,
                   gap_fit_exec, output_file, fit, verbose):
    """Fit a GAP with descriptors from  an .yml file"""

    # read properties from the param file
    with open(param_file) as yaml_file:
        params = yaml.safe_load(yaml_file)

    if atoms_filename is None:
        # not on command line, try in params
        if 'atoms_filename' in params:
            atoms_filename = params['atoms_filename']
        else:
            raise RuntimeError('atoms_filename not given in params file or as '
                               'command line input')
    else:
        if 'atoms_filename' in params:
            raise RuntimeError('atoms_filename given in params file and as '
                               'command line input')

    fitting_ci = ConfigSet_in(input_files=atoms_filename)

    if gap_file != 'GAP.xml':
        if params.get('gap_file', False):
            warnings.warn('overwriting gap_file from params with value from '
                          'command line')
        params['gap_file'] = gap_file

        if verbose:
            print("Overwritten param file's gap-filename from command")

    if output_file == 'default':
        output_file = os.path.splitext(params['gap_file'])[0] + '_output.txt'

    wfl.fit.gap_simple.run_gap_fit(fitting_ci, fitting_dict=params,
                                   stdout_file=output_file,
                                   gap_fit_exec=gap_fit_exec,
                                   do_fit=fit, verbose=verbose)


if __name__ == '__main__':
    cli()
