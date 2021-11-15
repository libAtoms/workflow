"""
This is a unified wrapper for Plane Wave DFT codes.

Currently implemented codes:
- CASTEP
- VASP
- Quantum Espresso
"""
from wfl.pipeline import iterable_loop
from wfl.calculators import castep, vasp, espresso


def evaluate_dft(
    inputs,
    outputs,
    calculator_name,
    base_rundir=None,
    dir_prefix=None,
    calculator_command=None,
    calculator_kwargs=None,
    output_prefix="DFT_",
    properties=None,
    keep_files="default",
    **kwargs,
):
    """

    Parameters
    ----------
    inputs: list(Atoms) / Configset_in
            input atomic configs, needs to be iterable
    outputs: list(Atoms) / Configset_out
        output atomic configs
    calculator_name: str {"CASTEP", "VASP", "QE"}
        name of Plane Wave DFT calculator, options are: "CASTEP", "VASP", "QE"
    base_rundir: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'DFT_'
            directory name prefix for calculations
    calculator_command: str
        command for calculator, only MPI and the executable
        eg. "mpirun -n 4 /path-to/castep.mpi"
    calculator_kwargs: dict
        all keyword arguments for the calculator object
    output_prefix : str / None
        prefix for info/arrays keys, None for SinglePointCalculator
        default is the calculator name
    properties : list(str), default None
            ase-compatible property names,
            None for default list (energy, forces, stress)
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            True : everything kept
            None, False : nothing kept, unless calculation fails
            "default"   : only ones needed for NOMAD uploads
                          depending on calculator
            list(str)   : list of file globs to save
    kwargs
        any other keyword arguments that need to be passed to the
        evaluation operation function.
        only implemented for VASP: potcar_top_dir, potcar_rel_dir

    Returns
    -------

    """
    # defaults
    if dir_prefix is None:
        dir_prefix = f"run_{calculator_name}_"

    # choose the calculator
    if calculator_name == "CASTEP":
        op = castep.evaluate_op
    elif calculator_name == "VASP":
        op = vasp.evaluate_op
    elif calculator_name == "QE":
        op = espresso.evaluate_op
    else:
        raise ValueError(f"Calculator name `{calculator_name}` not understood")

    # run the calculation in parallel
    return iterable_loop(
        iterable=inputs,
        configset_out=outputs,
        op=op,
        base_rundir=base_rundir,
        dir_prefix=dir_prefix,
        calculator_command=calculator_command,
        calculator_kwargs=calculator_kwargs,
        output_prefix=output_prefix,
        properties=properties,
        keep_files=keep_files,
        **kwargs,
    )
