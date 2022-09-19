"""
Collisions of molecules for reactivity search
"""

import os
import traceback
from tempfile import mkdtemp
import functools

import ase.io
import ase.io.extxyz
import numpy as np
try:
    import quippy
except ModuleNotFoundError:
    pass
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS

from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize import autoparallelize, iloop_docstring_post
from wfl.reactions_processing import trajectory_processing
from wfl.utils import vector_utils
from wfl.utils.parallel import construct_calculator_picklesafe


# noinspection PyPep8Naming
class Supercollider:
    """Collider class for collision of molecules

    Initial implementation by Hannes Margraf, further work by Tamas K. Stenczel

    Parameters
    ----------
    mol1, mol2 : ase.Atoms
        molecules to collide
    seed : str
        seed of the trajectory names
    cell : array_like
        cell to use in the calculation
    calc : ase.calculators.calculator.Calculator
        calculator to use for MD
    pot_filename : path_like
        file for quippy potential filename
    pot_arg_str : str
        arg_str for quippy potential
    dt : float, default=0.25
        time step of MD in fs
    T : float, default=1000.
        temperature in K
    velocity_params : tuple(float, float), default=
        parameters of velocity of collision as tuple of:
        (factor, const) which results in velocity of `factor * rand(0. to 1.) + const`
        uses ASE units
    pbc : bool|array_like, default=False
        periodic boundary conditions for calculation
    d0 : float, default=6.0
        distance of the molecules' CoM at the start
    trajectory_interval : int, default=1
        interval of printing trajectory files
    """

    def __init__(self, mol1, mol2, seed,
                 calc=None, pot_filename='GAP.xml', pot_arg_str='',
                 dt=0.25, T=1000., velocity_params=None, cell=(40., 20., 20.), pbc=False, d0=6.,
                 trajectory_interval=1):

        # calculator given or initialised from file
        if calc is not None:
            self.pot = calc
        else:
            self.pot = quippy.potential.Potential(pot_arg_str, pot_filename=pot_filename)

        self.T = T
        if velocity_params is not None:
            self.vel_params = velocity_params
        else:
            self.vel_params = [0.1, 0.15]

        self.mol1 = mol1.copy()
        self.mol2 = mol2.copy()
        self.d0 = d0

        self.cell = cell
        self.pbc = pbc
        self.seed = seed
        self.raw_traj_filename = "{}.traj".format(seed)
        self.mol_for_run = None
        self.trajectory_interval = int(trajectory_interval)

        self._relaxed_last_frame = None

        # dynamics setup
        self.dt = dt
        self.dyn = None
        self.traj = None

    def velocity_setup(self):
        factor, const = self.vel_params
        return factor * np.abs(np.random.randn(2)) + const

    def rotate_mol(self, mol, rotate_vec=None):
        # centering
        mol.translate(-mol.get_center_of_mass())
        # rotation
        if rotate_vec is None:
            rotate_vec = vector_utils.random_three_vector()
        mol.rotate((1.0, 0.0, 0.0), rotate_vec)
        # Set the internal momenta corresponding to T
        if len(mol) > 1:
            MaxwellBoltzmannDistribution(mol, self.T * units.kB)
            Stationary(mol)  # zero linear momentum
            ZeroRotation(mol)  # zero angular momentum
        else:
            mol.set_velocities(np.zeros((len(mol), 3)))

    def setup_atoms(self, distance=None):

        if not isinstance(distance, float):
            distance = self.d0

        self.rotate_mol(self.mol1)
        self.rotate_mol(self.mol2)
        self.mol2.translate((distance, 0.0, 0.0))

        # set collision velocity
        vel = self.velocity_setup()
        self.mol1.set_velocities(self.mol1.get_velocities() + (vel[0], 0.0, 0.0))
        self.mol2.set_velocities(self.mol2.get_velocities() + (-vel[1], 0.0, 0.0))

    def setup_supermolecule_vanilla(self, distance=4.):
        # molecules setup
        self.setup_atoms(distance=distance)

        # make supermolecule
        self.mol_for_run = self.mol1 + self.mol2
        self.mol_for_run.set_cell(self.cell)
        self.mol_for_run.set_pbc(self.pbc)
        self.mol_for_run.calc = self.pot

    def setup_dyn(self):
        self.dyn = Langevin(self.mol_for_run, self.dt * units.fs, temperature_K=self.T, friction=0.05)
        self.traj = Trajectory(self.raw_traj_filename, 'w', self.mol_for_run)
        self.dyn.attach(self.traj.write, interval=self.trajectory_interval)

    def run(self, nsteps=200):
        if self.mol_for_run is None:
            self.setup_supermolecule_vanilla()
        # in case mol_for_run has been set up separately, but dyn not
        if self.dyn is None:
            self.setup_dyn()

        # Run dynamics
        self.dyn.run(nsteps)

    def write_traj_xyz(self, fn=None, write_results=False, add_time=True, **kwargs):
        # convert traj
        frames = ase.io.read(self.raw_traj_filename, ':')
        if fn is None:
            fn = f"{self.seed}.raw_md.xyz"
        if add_time:
            self._add_time(frames)
        if self._relaxed_last_frame is not None:
            frames.append(self._relaxed_last_frame)
        ase.io.write(fn, frames, write_results=write_results, **kwargs)

    def _add_time(self, frames):
        # adding time to the info of the frames
        for i, at in enumerate(frames):
            at.info['time'] = i * self.dt

    def get_frames(self, choose=':', add_time=True, **kwargs):
        frames = ase.io.read(self.raw_traj_filename, choose, **kwargs)
        if add_time:
            self._add_time(frames)
        if self._relaxed_last_frame is not None:
            frames.append(self._relaxed_last_frame)
        return frames

    def relax_last_frame(self, fmax=0.05, steps=200, fn=None):
        # optimises the last frame and gives you the atoms object of the output
        at = self.get_frames('-1', add_time=False)
        at.calc = self.pot
        opt_mol = BFGS(at, trajectory="relax_trajectory.traj")
        opt_mol.run(fmax=fmax, steps=steps)

        # save the relaxation path in xyz
        ase.io.write("relax_trajectory.xyz", ase.io.read("relax_trajectory.traj", ":"))

        # for the reading of it to work later on
        at.info['time'] = -1
        self._relaxed_last_frame = at

        if fn is None:
            return at
        else:
            ase.io.write(fn, at, append=True)


def post_process_collision_autopara_wrappable(seed, calc,
                           *,
                           do_neb=False, do_ts_irc=False, force=True,
                           minim_interval=50, minim_kwargs=None, neb_kwargs=None, ts_kwargs=None, irc_kwargs=None):
    """Post processes a collision trajectory

    Implemented operations:
    - minimise every Nth frame (needed)
    - NEB + optional IRC+TS on changes of minimised structure

    Parameters
    ----------
    seed : iterable(str)
        job name seeds
    outputs: IGNORED
        ignored, output is written to seed.relax_*.xyz and seed.neb_*.xyz
    calc: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    do_neb : bool
    do_ts_irc : bool
    force : bool
        force the writing of the outputs
    minim_interval: int / str
        interval of frames to take for minimisation
    minim_kwargs: dict
    neb_kwargs: dict
        kwargs for neb calculators, not None triggers calculation
    ts_kwargs: dict
    irc_kwargs: dict
    {iloop_docstring_post}

    Returns
    -------

    """

    # pool-safe calculator creation
    calc = construct_calculator_picklesafe(calc)

    if isinstance(minim_interval, int):
        default_index = f"::{minim_interval}"
    elif isinstance(minim_interval, str):
        default_index = minim_interval
    else:
        raise TypeError("minima_interval has incorrect type, not str or int, type given:", type(minim_interval))

    # minimisation
    cfs_in_min = ConfigSet(input_files=f"{seed}.raw_md.xyz", default_index=default_index)
    cfs_out_min = OutputSpec(output_files=f"{seed}.relax_frames.xyz", force=force)
    cfs_inter = trajectory_processing.trajectory_min(
        configset=cfs_in_min, outputspec=cfs_out_min, calculator=calc,
        minimise_kwargs=minim_kwargs)

    # write the converged frames to file as well
    cfs_out_min_converged = OutputSpec(output_files=f"{seed}.relax_converged.xyz", force=force)
    for atl in cfs_inter.group_iter():
        if "config_type" in atl[-1].info.keys() and atl[-1].info['config_type'] == 'minim_last_converged':
            cfs_out_min_converged.write(atl[-1])
    cfs_out_min_converged.end_write()

    if do_neb and do_ts_irc:
        out_file_map = dict(neb_frames=f"{seed}.neb_frames.xyz",
                            neb_ts=f"{seed}.neb_ts.xyz",
                            neb_irc=f"{seed}.neb_irc.xyz")

        outputspec = OutputSpec(output_files=out_file_map, force=force)

        collected_images, collected_ts, collected_irc = trajectory_processing.trajectory_neb_ts_irc(
            cfs_inter, calc, neb_kwargs=neb_kwargs, ts_kwargs=ts_kwargs, irc_kwargs=irc_kwargs
        )

        outputspec.pre_write()
        for atoms_list in collected_images:
            outputspec.write(atoms_list, from_input_file="neb_frames")
        outputspec.write(collected_ts, from_input_file="neb_ts")
        for atoms_list in collected_irc:
            outputspec.write(atoms_list, from_input_file="neb_irc")
        outputspec.end_write()
    elif do_neb:
        outputspec_neb = OutputSpec(output_files=f"{seed}.neb_frames.xyz", force=force)
        trajectory_processing.trajectory_neb(configset=cfs_inter, outputspec=outputspec_neb, calculator=calc,
                                             neb_kwargs=neb_kwargs)
    elif do_ts_irc:
        raise ValueError("TS+IRC cannot be performed without having done NEB as well")


post_process_collision = functools.partial(autoparallelize, post_process_collision_autopara_wrappable)
post_process_collision.__doc__ = post_process_collision_autopara_wrappable.__doc__.format(iloop_docstring_post=iloop_docstring_post)


def run_pair(molecule1, molecule2, calc, seed="collision", nsteps=1000, **collision_kwargs):
    """Runs a single collision of a pair of molecules

    Parameters
    ----------
    molecule1, molecule2 : ase.Atoms
        molecules to collide
    calc : ase.calculator.Calculator
        ase-compatible calculator
    seed : str
        job name seed
    nsteps : int, default=1000
        number of MD steps to take
    collision_kwargs: dict
        passed to Supercollider init
    """

    if "seed" in collision_kwargs.keys():
        print(f"seed in collider_kwargs, while seed is set to {seed}\ncollider_kwargs:", collision_kwargs)
        del collision_kwargs["seed"]

    # 1. run collision
    collider = Supercollider(molecule1, molecule2, seed=seed, calc=calc, **collision_kwargs)
    collider.run(nsteps)
    collider.write_traj_xyz(write_results=True)


def run_collision_dir_management(indices, fragments, param_filename, rundir=None, **collide_kwargs):
    if rundir is None:
        rundir = os.getcwd()
    workdir = os.getcwd()  # this is where we are coming back at the end

    fmt_num = f"{indices[0]:0>2}-{indices[1]:0>2}"

    # create unique directory and run the collision
    tmp = mkdtemp(prefix=f'collision_{fmt_num}_', dir=rundir)
    print(f"Temporary directory is {tmp}")
    os.chdir(tmp)

    pot = quippy.potential.Potential('', param_filename=os.path.abspath(param_filename))

    try:
        run_pair(fragments[indices[0]], fragments[indices[1]], pot, **collide_kwargs)
    except Exception as e:
        print(f"Error encountered with collision at: {tmp}:")
        traceback.print_exc(e)
    finally:
        os.chdir(workdir)


parallel_collision = functools.partial(autoparallelize, run_collision_dir_management)


def multi_run_all_with_all(fragments, param_filename, workdir=None, min_atoms=0, num_repeat=1, excluded_formulas=None,
                           n_pool=None, **collide_kwargs):
    """Runs all pairs of collisions between fragments

    Parameters
    ----------
    fragments : list[ase.Atoms]
        list of fragments
    param_filename : str
        quippy potential file name
        assumes that init_args can be read from the xml
    workdir : path like, default=os.getcwd()
        working directory where to put the collision directories
    min_atoms : int, optional, default=0
        minimum number of atoms in a collision. Exclude any pair where the total number of atoms is below this
    excluded_formulas
        list of total formulas for which collision is not performed
    num_repeat : int, default=1
        number of repeated runs for each pair
    n_pool: int
        number of workers for parallel Pool
    collide_kwargs
        passed to collider
    """

    if workdir is None:
        workdir = os.getcwd()

    collision_indices = []

    n_fragments = len(fragments)
    for i in range(n_fragments):
        for j in range(i, n_fragments):
            fmt_num = f"{i:0>2}-{j:0>2}"

            # decide if collision is happening
            if len(fragments[i]) + len(fragments[j]) < min_atoms:
                print(f"Skipping pair {fmt_num} due to low number of total atoms")
                continue
            if excluded_formulas is not None:
                formula = (fragments[i] + fragments[j]).get_chemical_formula()
                if formula in excluded_formulas:
                    print(f"Skipping pair {fmt_num} due to excluded formula: {formula}")
                    continue

            [collision_indices.append((i, j)) for _ in range(num_repeat)]

    parallel_collision(collision_indices, None, fragments, param_filename, workdir, n_pool=n_pool, **collide_kwargs)
