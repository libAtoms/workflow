.. _command_line:

###############################
Command line interface
###############################

Description of all the CLI tools 

Automatically generated docummentation of all command line functions: :ref:`auto_command_line`


*******************************
Calculators
*******************************

You can perform DFT calculations on large sets of configurations with little effort within this workflow package. Currently ORCA, CASTEP and VASP are implemented. There are examples for these in the document, a few thoughts about the parallelism involved, and about how to contribute interfaces of other codes.



ORCA (outdated)
===============================

Use with ``wfl ref-method orca-eval [...]``

This is an advanced ORCA evaluator for radicals, using basin hopping in wavefunction space. However with appropriate settings, it behaves as a simple calculator doing one calculation per structure.

Note, that the calculations can benefit from having access to fast storage, for example SSD scratch drives. Specifying a fast scratch path in the ``-tmp, --scratch-path`` argument lets you use this, where temporary directories will be created and only saved (to cwd) if the calculations fail.

Example of setting for advanced calculation:

.. code-block:: console
  
  wfl -v ref-method orca-eval -o frames_dft.xyz --scratch-path="/scratch/" --orca-command="/path/to/orca" \
    -n 3 -p 32 --kw="n_hop=15" structures_in.xyz 

Example of a single calculation per configuration:

.. code-block:: console 

  wfl -v ref-method orca-eval -o frames_dft.xyz --scratch-path="/scratch/" --orca-command="/path/to/orca" \
    -n 1 -p 32 --kw="n_hop=1" structures_in.xyz

This calculator needs you to explicitly set the number of processes to use, serial ORCA calculations are used, unless you specify in the appropriate block.

full help:

.. code-block:: console 

  $ wfl ref-method orca-eval --help
  Usage: wfl ref-method orca-eval [OPTIONS] [INPUTS]...

  Options:
    --output-file TEXT             [required]
    --output-all-or-none
    --output-prefix TEXT           prefix in info/arrays for results
    --base-rundir TEXT             directory to put all calculation directories
                                   into

    --directory-prefix TEXT
    --calc-kwargs, --kw TEXT       Kwargs for calculation, overwritten by other
                                   options

    --keep-files TEXT              How much of files to keep, default is NOMAD
                                 compatible subset

    --orca-command TEXT            path to ORCA executable, default=`orca`
    -tmp, --scratch-path TEXT      Directory to use as scratch for calculations,
                                   SSD recommended, default: cwd

    -nr, --n-run INTEGER           Number of global optimisation runs for each
                                   frame  [required]

    -nh, --n-hop INTEGER           Number of hopping steps to take per run
                                 [required]

    --orca-simple-input TEXT       orca simple input line, make sure it is
                                   correct, default is recPBE with settings
                                   tested for radicals

    --orca-additional-blocks TEXT  orca blocks to be added, default is None
    --help                         Show this message and exit.


CASTEP
===========================================


The CASTEP calculator wrapper can perform simple energy, force and stress calculations.

The default settings can be simply used with:

.. code-block:: console 

  wfl ref-method castep-eval --output-file structures_dft.xyz properties="energy forces stress" structures_in.xyz


Including some settings to the calculator:

.. code-block:: console 

  wfl ref-method castep-eval --output-file structures_dft.xyz properties="energy forces stress" \
    --castep-kwargs="ecut=500 xc=pbesol" structures_in.xyz

Full help

.. code-block:: console 

  $ wfl ref-method castep-eval --help
  Usage: wfl ref-method castep-eval [OPTIONS] [INPUTS]...

  Options:
    --output-file TEXT       [required]
    --output-all-or-none
    --output-prefix TEXT     prefix in info/arrays for results
    --base-rundir TEXT       directory to put all calculation directories into
    --directory-prefix TEXT
    --pp-path TEXT           PseudoPotentials path
    --properties TEXT        properties to calculate, string is split
    --castep-command TEXT    command, including appropriate mpirun
    --castep-kwargs TEXT     CASTEP keywords, passed as dict
    --keep-files TEXT        How much of files to keep, default is NOMAD
                             compatible subset

    --help                   Show this message and exit.
  
VASP
==============================================

The VASP Calculator wrapper is based on ASE's ``Vasp2``, and can do energy, force, and stress calculations.  The command will come from the ``VASP_COMMAND`` (``VASP_COMMAND_GAMMA`` for input configs with ``Atoms.pbc == [False] * 3``) unless the ``--vasp-command`` argument is passed.

The default way for it to find PAWs is somewhat different from the normal behavior for ``Vasp2``. The normal way is the ``VASP_PP_PATH`` environment variable, followed by a subdirectory which depends on XC functional. The argument ``--potcar-top-dir`` overrides ``VASP_PP_PATH`` if specified. The subdirectory defaults to ``.``, unless overridden by the ``--potcar-rel-dir``, instead of using the normal XC-dependent heuristic.  The PAW for each element is normally found in ``<chem_symbol>/POTCAR`` below whatever directory is chosen (the ``setups`` keyword argument can contain a dict that sets a suffix to the ``<chem_symbol>`` (e.g. ``_pv`` for p-electrons in valence) for each atomic number or chemical symbol key, but there is no interface to pass such a dict yet).

The simplest way to use it is by using parameters from VASP's normal input files:

.. code-block:: console 

  wfl ref-method vasp-eval --output-file structures_dft.xyz --incar INCAR --kpoints KPOINTS structures_in.xyz

Particular keywords can be set (or override those read from INCAR), lile 

.. code-block:: console 

  wfl ref-method vasp-eval --output-file structures_dft.xyz --incar INCAR --incar-dict "encut=500" \
      --kpoints KPOINTS structures_in.xyz


Full help (keyword help messages coming soon)

.. code-block:: console 

  $ wfl ref-method vasp-eval --help
  Usage: wfl ref-method vasp-eval [OPTIONS] [INPUTS]...

  Options:
    --output-file TEXT     [required]
    --output-all-or-none
    --base-rundir TEXT
    --output-prefix TEXT
    --properties TEXT
    --incar TEXT
    --kpoints TEXT
    --incar-dict TEXT
    --potcar-top-dir TEXT
    --potcar-rel-dir TEXT
    --vasp-command TEXT
    --help                 Show this message and exit.


.. toctree::
   :maxdepth: 2

   command_line.automatic_docs.rst
