# NOT REALLY A FULL CLI, JUST ROUTINES EXTRACTED FROM wfl/cli/cli.py

from wfl.reactions_processing import trajectory_processing

@subcli_generate_configs.command('remove-sp3-Hs')
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--outputs", "-o", help="Output filename, see Configset for details", required=True)
@click.option("--force", "-f", help="force writing", is_flag=True)
def configs_from_smiles(ctx, inputs, outputs, force):
    """Removes all sp3 hydrogen atoms for given molecules"""

    verbose = ctx.obj["verbose"]

    inputs = ConfigSet(input_files=inputs)
    outputs = OutputSpec(output_files=outputs, force=force)

    if verbose:
        print(inputs)
        print(outputs)

    user.generate.radicals.abstract_sp3_hydrogen_atoms(inputs=inputs, outputs=outputs)


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

    wfl.generate.user.generate.collision.multi_run_all_with_all(
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

    wfl.generate.user.generate.collision.post_process_collision(
        seed=seeds, calc=calc, do_neb=do_neb, do_ts_irc=do_ts_irc, minim_kwargs=minim_kwargs,
        minim_interval=minim_interval, neb_kwargs=neb_kwargs,
        ts_kwargs=ts_kwargs, irc_kwargs=irc_kwargs, n_pool=n_pool,
    )


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

    configset = ConfigSet(input_files=inputs)
    outputspec = OutputSpec(output_files=outputs, force=force)
    if verbose:
        print(configset)
        print(outputspec)
        sys.stdout.flush()

    # initialisations for calculation
    frames_train = ase.io.read(train_filename, ":")
    soap_dict, desc_ref = trajectory_processing.create_desc_ref(frames_train, cutoff_list, z_list)
    if verbose:
        print("Calculated SOAP vectors for training set")

    for at in configset:
        at = trajectory_processing.calc_max_similarity_atoms(at, soap_dict, desc_ref)
        outputspec.write(at, from_input_file=configset.get_current_input_file())

    outputspec.end_write()


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

    configset = ConfigSet(input_files=inputs,
                                default_index=(f"::{stride}" if stride is not None else ":"))
    if cut_threshold is not None:
        # cutting by global SOAP metric -- simply recreating the configset with indices calculated
        new_inputs = []

        for subcfs in configset.group_iter():
            idx = trajectory_processing.cut_trajectory_with_global_metric(subcfs, cut_threshold)
            current_fn = configset.get_current_input_file()
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
        configset = ConfigSet(input_files=new_inputs)

    outputspec = OutputSpec(output_files=output, force=force)

    if verbose:
        print(configset)
        print(outputspec)
        sys.stdout.flush()

    z_list = []
    num_dict = dict()
    for z, num in n_select:
        z_list.append(z)
        num_dict[z] = num
    if verbose:
        print("(z, num) to take:", num_dict)

    weighted_cur.selection(configset, outputspec, z_list, descriptor, limit, num_dict)
    outputspec.end_write()
