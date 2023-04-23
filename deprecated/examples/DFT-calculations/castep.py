"""
This is a simple example of how to use Quantum Espresso
"""
from pprint import pprint

from wfl.configset import ConfigSet, OutputSpec
from wfl.utils.logging import print_log
from wfl.calculators.dft import evaluate_dft


def main(verbose=True):
    # settings
    # replace this with your local configuration in productions
    workdir_root = "CASTEP-calculations"
    castep_kwargs = {
        "ecut": 400.0,
        "kpoint_mp_spacing": 0.1,
        "xc": "pbesol",
        "SPIN_POLARIZED": False,
        "PERC_EXTRA_BANDS": 100,
        "MAX_SCF_CYCLES": 200,
        "NUM_DUMP_CYCLES": 0,
        "MIXING_SCHEME": "pulay",
        "MIX_HISTORY_LENGTH": 20,
        "SMEARING_WIDTH": 0.2,
        "FIX_OCCUPANCY": False,
    }
    castep_command = "mpirun -n 2 castep.mpi"

    # IO
    configs_in = ConfigSet(input_files="periodic_structures.xyz")
    configs_out = OutputSpec(
        output_files="DFT_evaluated.CASTEP.periodic_structures.xyz",
        force=True,
        all_or_none=True,
    )

    if verbose:
        print_log("Quantum Espresso example calculation")
        print(configs_in)
        print(configs_out)
        print(f"workdir_root: {workdir_root}")
        print(f"castep_command: {castep_command}")
        pprint(castep_kwargs)

    # run the calculation
    _ = evaluate_dft(
        calculator_name="CASTEP",
        inputs=configs_in,
        outputs=configs_out,
        workdir_root=workdir_root,  # directory where to put the calculation directories
        calculator_command=castep_command,
        calculator_kwargs=castep_kwargs,
        keep_files="default",  # keeps the .pwo file only
    )


if __name__ == "__main__":
    main()
