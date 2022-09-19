"""
This is a simple example of how to use Quantum Espresso
"""
import os
import pathlib
from pprint import pprint

import requests

from wfl.calculators.dft import evaluate_dft
from wfl.configset import ConfigSet, OutputSpec
from wfl.utils.logging import print_log

PSPOT_DOWNLOAD_MAP = [
    ("Si", "https://www.quantum-espresso.org/upf_files/Si.pbe-n-kjpaw_psl.1.0.0.UPF",),
    ("C", "https://www.quantum-espresso.org/upf_files/C.pbe-n-kjpaw_psl.1.0.0.UPF"),
]


def download_pseudo_potentials(directory, overwrite=False, verbose=True):
    """
    Example of downloading pseudo potentials, in order to have the calculation run out of the box.
    There is no guarantee that these are the correct ones for your system and problem at hand.
    """
    pseudos = dict()

    for element, url in PSPOT_DOWNLOAD_MAP:
        filename = f"{element}.UPF"
        pseudos[element] = filename

        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path) and not overwrite:
            # skip download
            if verbose:
                print_log(f"QE pseudo potential found for {element}, skipping download")
            continue

        # make sure dir exists
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        # download
        try:
            r = requests.get(url)
        except ConnectionError:
            # no internet!
            raise ConnectionError(
                "No Internet for Quantum Espresso pseudo-potential download."
            )

        # test the result
        if r.status_code != requests.codes.ok:
            raise RuntimeError("Quantum Espresso pseudo-potential download failed")

        # write to file
        with open(full_path, "w") as file:
            file.write(r.text)

        if verbose:
            print_log(f"QE pseudo potential downloaded for {element} into {full_path}")

    return pseudos


def main(verbose=True):
    # settings
    # replace this with your local configuration in productions
    pspot_dir = os.path.join(os.getcwd(), "QE-pseudo-potentials")
    pspot_map = download_pseudo_potentials(pspot_dir, verbose=verbose)
    workdir_root = "QE-calculations"
    qe_command = "mpirun -n 2 pw.x"  # local command for pw.x, with MPI if needed

    # IO
    configs_in = ConfigSet(input_files="periodic_structures.xyz")
    configs_out = OutputSpec(
        output_files="DFT_evaluated.QuantumEspresso.periodic_structures.xyz",
        force=True,
        all_or_none=True,
    )

    # Settings. These are very minimal, change it as needed
    settings = dict(
        pseudopotentials=pspot_map,
        input_data={"SYSTEM": {"ecutwfc": 50, "input_dft": "PBE",}},
        pseudo_dir=pspot_dir,
        kpts=(2, 2, 2),
    )

    if verbose:
        print_log("Quantum Espresso example calculation")
        print(configs_in)
        print(configs_out)
        print(f"workdir_root: {workdir_root}")
        print(f"qe_command: {qe_command}")
        pprint(settings)

    # run the calculation
    _ = evaluate_dft(
        calculator_name="QE",
        inputs=configs_in,
        outputs=configs_out,
        workdir_root=workdir_root,  # directory where to put the calculation directories
        calculator_command=qe_command,  # local command for pw.x, with MPI if needed
        calculator_kwargs=settings,
        keep_files="default",  # keeps the .pwo file only
    )


if __name__ == "__main__":
    main()
