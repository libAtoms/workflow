# NOT A REAL CLI, JUST A STUB WITH ROUTINES REMOVE FROM wfl/cli/cli.py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wfl.plotting import reactions_plotting, plot_ef_correlation, plot_2b
from wfl.plotting import normal_modes

@cli.group("plot")
@click.pass_context
def subcli_plot(ctx):
    pass

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
    frames = ConfigSet(input_files=inputs)
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
