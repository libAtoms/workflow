"""
This is a simple example of how to use VASP
"""
import os
from pprint import pprint

from wfl.calculators.dft import evaluate_dft
from wfl.configset import ConfigSet, OutputSpec
from wfl.utils.logging import print_log


def main(verbose=True):
    # settings
    # replace this with your local configuration in productions
    workdir_root = "VASP-calculations"
    vasp_kwargs = {
        "encut": 200.0,
        "kspacing": 0.3,
        "xc": "pbesol",
        "nelm": 200,
        "sigma": 0.2,
    }
    os.environ["MPIRUN_EXTRA_ARGS"] = "-np 2"
    vasp_command = "vasp.para"

    # path for your pseudo-potential directory
    assert "VASP_PP_PATH" in os.environ

    # IO
    configs_in = ConfigSet(input_files="periodic_structures.xyz")
    configs_out = OutputSpec(
        output_files="DFT_evaluated.VASP.periodic_structures.xyz",
        force=True,
        all_or_none=True,
    )

    if verbose:
        print_log("VASP example calculation")
        print(configs_in)
        print(configs_out)
        print(f"workdir_root: {workdir_root}")
        print(f"vasp_command: {vasp_command}")
        pprint(vasp_kwargs)

    # run the calculation
    _ = evaluate_dft(
        calculator_name="VASP",
        inputs=configs_in,
        outputs=configs_out,
        workdir_root=workdir_root,  # directory where to put the calculation directories
        calculator_command=vasp_command,
        calculator_kwargs=vasp_kwargs,
        keep_files="default",  # keeps files minimum for NOMAD upload
    )


if __name__ == "__main__":
    main()
