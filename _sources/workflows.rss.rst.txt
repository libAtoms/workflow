.. _rss:


###################################
How to do GAP-RSS
###################################

You would like to explore the structural space of an element/alloy and gain a GAP potential at the same time? Here is a small introduction of how to do this in an automated manner with this repository.

All you need to do is choose your favourite elements, follow the instructions below and run the workflow, then test, use and expand your model as you like.


**********************************
Before you start
**********************************


Make a new directory and create the following files:
- ``length_scales.json`` defining the bond lengths and typical volumes for the elements
- ``config.json`` this is the main configuration file
- fitting settings, in json format (see below)
- some job script for your cluster/workstation

The length scales are as follows:

.. code-block:: python

  {
    "3": {"bond_len": [3.0, "NB gap-rss"],
          "min_bond_len": [2.4, "NB gap-rss"],
          "other links": {},
          "vol_per_atom": [20.0, "NB gap-rss"]},
   [...]
  }


Where the primary keys are the atomic numbers (as strings, as it is a json file), and typical bond length, minimal bond length and vol per atom are defined as floats and some comment. Additionally any keys can be added for other information, which is not parsed.

The fitting settings are as follows:

.. code-block:: python 

  { "stages" : [
     { "error_scale_factor" : 10.0, "add_species" : "manual_Z_pair",
         "descriptors" : [ { "desc_str" : "distance_Nb order=2 cutoff=${BOND_LEN_Z1_Z2*2.5} cutoff_transition_width=${BOND_LEN_Z1_Z2*2.5/5.0} compact_clusters Z={{${Z1} ${Z2}}}",
                             "fit_str" : "n_sparse=15 covariance_type=ard_se theta_uniform=${BOND_LEN_Z1_Z2*2.5/5.0} sparse_method=uniform f0=0.0 add_species=F",
                             "count_cutoff" : "_F_ ${BOND_LEN_Z1_Z2*1.4}" } ] } ,
     { "error_scale_factor" : 1.0, "add_species" : "manual_universal_SOAP",
         "descriptors" : [ { "desc_str" : "soap n_max=12 l_max=6 atom_sigma=${ATOM_SIGMA} cutoff=${R_CUT} cutoff_transition_width=${R_TRANS} central_weight=1.0 Z=${Zcenter} n_species=${nZ} species_Z={{${Zs}}}",
                             "fit_str" :  "n_sparse=1000 f0=0.0 covariance_type=dot_product zeta=4 sparse_method=cur_points print_sparse_index add_species=F" } ] }
    ],
    "gap_params" : "default_sigma='{0.0025 0.0625 0.125 0.125}' sparse_jitter=1.0e-8 do_copy_at_file=F sparse_separate_file=T"
  }

This is a file containing ``stages`` of GAP fitting, which are descriptors laid out for fitting hierarchically. From first to last, a model will be created for each and the energy scale (``delta``, typical variance of GP) for the next one defined from the energy variance per descriptor.
Additionally, general settings can be given in ``gap_params`` for the fit.

The filename of this file goes into the main config's ``fit/GAP_template_file`` field.


Main config
===================================

This is also a json file, with settings of how to execute certain steps of the GAP-RSS workflow. For the structure and keywords in the file please take a look at the examples. Here, we are giving a small overview of what the workflow is doing and what you can tweak in it.

The structure and field names may change during development, input file structure may be updated later on.


***********************************
Steps of GAP-RSS
***********************************

The GAP-RSS workflow consists of a few steps, which can be accessed with given command line options. Here is a summary of them and then more detailed overview of what they are doing.

The central command line interface for iterative GAP-RSS is ``gap_rss_iter_fit``, which has all the subcommands defined. If you are interested in the implementation, dig into it at ``wfl/cli/gap_rss_iter_fit.py`` where these are defined.

The steps:
1. Preparation: check and process input files
2. initial step: gather initial data, fit a model on it
3. rss steps: RSS with latest model, selection and fitting
4. MD step: MD sampling of bulk defects, selection and fitting


************************************
Preparation step
************************************

Use with ``gap_rss_iter_fit --configuration <main config json> prep``

This step is reading the input files, summarises the compositions defined, created ``buildcell`` input files for them, and simple dimer and isolated atom configurations.

The compositions are defined as tuples of formula unit and relative weight of them. These are normalised and the random structures are created with respective distribution.

Buildcell inputs are created for both narrow and wide volume ranges, based on the given volume per atom targets from the input. These are separated by the formulas and for some between even and off number of formula units in the unit cell, which is because of the general observation in crystallography that even number of formula units are much more common than odd ones. The even and odd ones are sampled in a 4:1 ratio later on.

Additionally the selection settings and the buildcell info are written to respective yaml files and a json file is created by processing the fitting settings, stages and descriptors with the  elements and their length scales specified. Finally the bare dimer and single atom configurations are written, DFT energy and forces will be evaluated on them in the initial step.


****************************************
DFT settings
****************************************

The main config has a section of ``DFT_evaluate`` specifying the settings for reference calculations. Currently VASP and CASTEP are implemented as calculators, specify which one you are using in the ``DFT_evaluate/calculator`` field with upper case name.

Note: the ``/`` refers to the hierarchy of the json file's dictionaries, so ``foo/bar`` refers to the ``bar`` field in the ``foo`` dict.

The keywords passed to the calculator are in ``DFT_evaluate/kwargs``, please refer to the calculator for details of these and physical/modeling knowledge of the values you want to set.

Additionally any keys can be given in ``DFT_evaluate``, of which only some are read, specific to calculators. Eg. if using CASTEP, the ``DFT_evaluate/CASTEP_pp_path`` is used as the pseudopotential directory.


****************************************
Config selection
****************************************

Currently two algorithms are implemented: CUR and greedy FPS. Both are using global descriptors and have no information of the previously selected set, only try to give the "best" from what they are given from the current step.

The descriptor needs to be specified in ``global/config_selection_descriptor``, where it is needed to use the ``average`` so that quippy is creating the global descriptor. Atomic types and length are adjusted automatically, use ``global/config_selection_descriptor_add_species=manual_Zcenter`` for this.

****************************************
Initial step
****************************************

Use with ``gap_rss_iter_fit --configuration <main config json> initial_step``

This is the first step for model building, where a preliminary GAP model needs to be created. The way this is done here is by creating random structures (``initial_step/buildcell_total_N``) with buildcell and sampling some (``initial_step/by_desc_select_N``) from them for DFT evaluation.

Additionally, the isolated atoms and dimer structures are evaluated and added to the fit for e0 definition and core repulsion.


****************************************
RSS step
****************************************

Use with ``gap_rss_iter_fit --configuration <main config json> rss_step``

These are the main steps for learning the energy landscape of crystals and minima. Just as in AIRSS, random structures are created and relaxed, however here this is done with the latest GAP model. Pre-selection is carried out with flat histogram on the minima and on the trajectories, so descriptors are only calculated for a subset of the structures. Then descriptor based selection is performed.

Minimisation can be done with pressure applied, which is defined in the ``rss_step/minim_pressure`` field, as one of the following:
- float: fixed pressure in GPa
- ["exponential", float] exponential distribution, rate=1. and scaled by float given
- ["normal_positive", mean, sigma]: normal distribution with (mean, sigma) thrown away if negative value drawn, max 1000 tries
- ["uniform", lower, upper]: uniform distribution between bounds (lower, upper)

In practice, this is first on the minima with parameters ``rss_step/minima_flat_histo_N`` and ``rss_step/minima_by_desc_select_N``.
If ``rss_step/select_convex_hull`` is true, then the structures lying on the convex hull are preselected as well, but not their  trajectories.
Then the trajectories of the minima are used for a new flat histogram and descriptor based sampling, where ``rss_step/selected_traj_flat_histo_N`` and ``rss_step/selected_traj_by_desc_select_N`` are the parameters. The latter is the number of structures actually selected from the generated set and this many DFT calculations will be carried out.

Additionally, the temperature of the flat histogram sampling can be specified, as an iteration specific value under ``rss_step/iter_specific/flat_histo_kT`` with keys of iteration number and values. These may be useful for the first few steps, setting somewhat larger value to sample the higher energy configs as well.


****************************************
MD step
****************************************

Use with ``gap_rss_iter_fit --configuration <main config json> MD_bulk_defect_step``

This is performing MD with bulk defects, aimed to sampling rather than physically correct experiments.

The initial structure to create defects in can be specified with the ``--minima_file`` parameter, or is omitted then the same procedure is repeated as in an RSS step, but the final structures are not evaluated with DFT, but used as initial structures here. See parameters of the RSS step above.

From the initial structures, supercells are built for up to four different types of MD: bulk, vacancy, interstitial and surface. The number of runs to do with each are controlled by ``MD_bulk_defect_step/N_<name>`` fields respectively. The supercells will have up to ``MD_bulk_defect_step/max_n_atoms`` number of atoms in them.

For the building of the surfaces, the thickness of matter and vacuum can be given as: ``MD_bulk_defect_step/surface_min_thickness`` and ``MD_bulk_defect_step/surface_vacuum`` both defaulting to 8.0 Å if none is given.

For the dynamics, the time step (``MD_bulk_defect_step/MD_dt``) can be specified applying to all, and the temperature, as well as the number of steps can be set for the bulk and the defect MDs separately. The temperature can be a given value, or a range of two floats with an optional integer for the number of stages to use in the ramp.

Finally selection is carried out with pre-selection on flat histogram (``MD_bulk_defect_step/final_flat_histo_N``) and then by descriptor (``MD_bulk_defect_step/final_by_desc_select_N``) to yield the structures for DFT evaluation.


****************************************
Job script and runtime settings
****************************************

Having all of the configs, we only need to submit a job to a cluster and carry out the calculations. Here are a few ideas and hints of how to do this, as well as settings for runtime that will be needed.

Take a node / workstation that has enough memory for the ``gap_fit`` steps with the size of dataset expected to be generated.

The code is parallelised with python's ``multiprocessing.Pool`` at many steps, for example minimisation, DFT evaluations and descriptor calculations.

The number of workers to use in the pool is controlled by the ``WFL_NUM_PYTHON_SUBPROCESSES`` env variable. Set this to the number of physical cores, hyperthreading is not expected to be beneficial.

.. code-block:: console

  # usable logical threads are given in NSLOTS
  export WFL_NUM_PYTHON_SUBPROCESSES=${NSLOTS}

Export the DFT calculator's executable, as ASE can understand it:

.. code-block:: console 

  # is using CASTEP
  # mpi or serial, makes no difference
  export CASTEP_COMMAND=/path/to/castep.mpi 
  
  # if using VASP - both default and gamma-point ones are needed
  export VASP_COMMAND=/path/to/vasp.serial
  export VASP_COMMAND_GAMMA=/path/to/vasp.gamma_serial

Export the installation of buildcell, ``GRIF`` is the prefix for \**g**ap_**r**ss_**i**ter_**f**it

.. code-block:: console 

  export GRIF_BUILDCELL_CMD=$HOME/programs_womble0/airss-0.9.1/bin/buildcell

Set the number of OMP threads to 1 in general and to the maximum for gap_fit

.. code-block:: console 

  export OMP_NUM_THREADS=1
  export WFL_GAP_FIT_OMP_NUM_THREADS=${NSLOTS}

The active iteration's number is written in the file ``ACTIVE_ITER``, which if you keep then the iteration number will be increased. Directories will be created with names ``run_iter_<number>`` and all work of a given iteration self contained in them.

Anything that is done already can be skipped, the ``OutputSpec`` implementation is taking care of this, which lets you not repeat work in a lot of cases when restarting the calculations.

An efficient and handy way to run multiple iterations is as follows:

.. code-block:: bash

  # remove leftover files
  for f in run_iter_* ACTIVE_ITER atoms_and_dimers.xyz gap_rss.*.out; do
    echo "WARNING: Trace of old run found file '$f'" 1>&2
    break
  done

  # start the numbering from zero, will skip ones done
  rm -f ACTIVE_ITER
  
  # abort if any fail
  set -e
  
  # set the config json file, and number of total steps
  system_json=main_config_file.json
  total_rss_steps=24
  
  # prep
  gap_rss_iter_fit -c ${system_json} prep >> gap_rss.prep.out
  
  # RSS iters
  gap_rss_iter_fit -c ${system_json} initial_step >> gap_rss.initial_step.out
  for iter_i in $(seq 1 ${total_rss_steps}); do
    gap_rss_iter_fit -c ${system_json} rss_step >> gap_rss.rss_step.${iter_i}.out
  done
  
  # do one MD steps, or more if you want with a loop
  # bulk/defect supercell MD iter
  iter_i=${total_rss_steps}+1
  gap_rss_iter_fit -c ${system_json} MD_bulk_defect_step >> gap_rss.${iter_i}.MD_step.out
  


