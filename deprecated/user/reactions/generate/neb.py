from ase.neb import NEB
from ase.optimize import BFGS

from user.generate import ts
from user.generate.irc import calc_irc


def neb_generic(start, end, calculator, nimages=17, interpolation_method="idpp", fmax=None, steps=None, k=None,
                remove_rotation_and_translation=True, verbose=False):
    """Generic NEB calculation, relaxed end structures are required

    Parameters
    ----------
    start: ase.Atoms
    end: ase.Atoms
    calculator: ase.calculators.Calculator
    nimages: int
        number of images, including the ends of the band
    interpolation_method: {'idpp', 'linear', None}
        use interpolation with given method
        None triggers no interpolation
    fmax: float
        force max for optimisation
    steps: int
        max number of steps for optimisation
    remove_rotation_and_translation : bool
        NEB parameter, minimise rotations and translations of the entire structure at each step
    k : float
        string parameter for NEB
    verbose: bool, default False
        optimisation logs are not printed unless this is True

    Returns
    -------
    images: list(ase.Atoms)
        relaxed images
    """
    if verbose:
        logfile = '-'
    else:
        logfile = None

    images = neb_create_initial_path(start, end, nimages, calculator, remove_rotation_and_translation,
                                     interpolation_method)

    neb = NEB(images, k=k, remove_rotation_and_translation=remove_rotation_and_translation,
              allow_shared_calculator=True)
    opt = BFGS(neb, logfile=logfile)
    opt.run(fmax=fmax, steps=steps)

    return images


def neb_create_initial_path(start, end, nimages, calculator, remove_rotation_and_translation, interpolation_method):
    """NEB initial path creation from start and end frames

    Currently only using the images and later interpolating between them

    Parameters
    ----------
    start
    end
    nimages
    calculator
    remove_rotation_and_translation
    interpolation_method

    Returns
    -------

    """
    images = [start]
    for _ in range(nimages - 2):
        images.append(start.copy())
    images.append(end)

    # interpolation with NEB's method
    if interpolation_method in ["linear", "idpp"]:
        neb_for_interpolation = NEB(images, remove_rotation_and_translation=remove_rotation_and_translation,
                                    allow_shared_calculator=True)
        neb_for_interpolation.interpolate(method=interpolation_method)
    elif interpolation_method:
        # this filters None, False, etc that evaluated to False
        print("NEB interpolation skipped due to unknown interpolation method given:", interpolation_method)

    for at in images:
        at.calc = calculator

    return images


def neb_with_ts_and_irc(start, end, calculator, neb_kwargs=None, ts_kwargs=None, irc_kwargs=None):
    """NEB calculation and TS+IRC from the transition found

    Parameters
    ----------
    start: ase.Atoms
    end: ase.Atoms
    calculator: ase.calculators.Calculator
    neb_kwargs: dict
    ts_kwargs: dict
    irc_kwargs: dict

    Returns
    -------
    neb_images: list(ase.Atoms)
        converged NEB images
    ts_structure: ase.Atoms
        TS optimised from the transition found in NEB
    irc_ends: list(ase.Atoms)
        ends of IRC calculation both forward and backward, ie. the two minima connected by the TS

    """
    if neb_kwargs is None:
        neb_kwargs = dict()
    if ts_kwargs is None:
        ts_kwargs = dict()
    if irc_kwargs is None:
        irc_kwargs = dict()

    neb_images = neb_generic(start, end, calculator, **neb_kwargs)

    def _key(x):
        return x.info["energy"]

    # ts calculation
    ts_candidate = max(neb_images, key=_key).copy()
    ts_traj = ts.calc_ts(ts_candidate, calculator, **ts_kwargs)
    ts_structure = ts_traj[0][-1]  # nested list of list

    # irc, only the ends of it are needed
    irc_traj = calc_irc(ts_structure.copy(), calculator, **irc_kwargs)
    irc_ends = [irc_traj[0][0], irc_traj[0][-1]]

    return neb_images, ts_structure, irc_ends
