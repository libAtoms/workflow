import os
import shutil
import sys
import tempfile
import glob, sys, pathlib
#sys.path.appned("/work/global_optimization")
#from helpers import prepare_input, apply_constraints

import numpy as np
from ase.optimize.minimahopping import MinimaHopping
from ase.io.trajectory import Trajectory
from ase.io import read

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.misc import atoms_to_list
from wfl.generate.utils import config_type_append
from wfl.utils.parallel import construct_calculator_picklesafe


def _get_MD_trajectory(rundir):
	
	md_traj = []
	mdtrajfiles = sorted([file for file in glob.glob(f"{rundir}/md*.traj")])
	for mdtraj in mdtrajfiles:
		for at in read(f"{mdtraj}@:"):
			config_type_append(at, 'traj')
			md_traj.append(at)

	return md_traj


def _get_after_explosion(rundir):
	optimizations = sorted(glob.glob(f"{rundir}/qn*.traj"))[-1]
	MDs = glob.glob(f"{rundir}/md*.traj")
	traj = []

	if os.path.isfile(f"{rundir}/minima.traj") and len(read(f"{rundir}/minima.traj@:")) >= 3: 	
		print("exploded, but at least 3 minima found", flush=True)
		minima = read(f"{rundir}/minima.traj@:")
		for at in minima:
			config_type_append(at, 'minima')
		traj += minima
		traj += _get_MD_trajectory(rundir)	
		
		return traj

	elif len(MDs) == 0:
		print("No MD trajectory obtained: Exploded during optimization", flush=True)
		for opt_traj in Trajectory(optimizations):
			config_type_append(opt_traj, "hopping_traj")
			traj.append(opt_traj)

		return traj

	elif len(MDs) != 0:
		print("Exploded during MD simulation", flush=True)
		
		return None #_get_MD_trajectory(rundir)	

def print_minhop_parameter(hop):

    print("="*40)
    print("Minima Hopping parameters")
    print("="*40)
    for key in hop._default_settings.keys():
        print(f"{key:<16s} :  {getattr(hop, '_' + key)}")
    print("="*40)

#def contingency_plan():
#        count = 0
#        num_minima = len([atoms for atoms in list(out_config.to_ConfigSet()) if atoms.info["config_type"].split("_")[-1] == "minima"])
#        while num_minima < 3:
#
#            print("Minima from minhop are less than 3! Ten more hops are added")
#            os.system("rm -f hop.log md* qn* ")
#            atoms = prepare_input(smiles=smiles, facet=facet,randseed=count+100, return_configset=False)#, coadsorbate=coadsorbate, num_coadsorbate=num_coadsorbate)
#
#           if num_coadsorbate == 0: 
#               apply_constraints(atoms,
#                       relax_metal=relax_metal,
#                       smiles=smiles)  
#           else:
#               apply_constraints(atoms,
#                       relax_metal=relax_metal,
#                       constrain_Hookean = True,
#                       method = None)
#            print("initial structure for minima hopping changed!")
#
#            if crude_gap:
#                kwargs["totalsteps"] += 10
#                kwargs["Ediff"] = 6
#				opt = MinimaHopping(atoms, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold,
#								mdmin=mdmin, fmax=fmax, timestep=timestep, **opt_kwargs)
#				opt(totalsteps=totalsteps)
#            else:
#                kwargs["Ediff"] = 6
#				opt = MinimaHopping(atoms, Ediff0=6, T0=T0, minima_threshold=minima_threshold,
#								mdmin=mdmin, fmax=fmax, timestep=timestep, **opt_kwargs)
#				opt(totalsteps=totalsteps)
##               hop_re = MinimaHopping(atoms, Ediff0=6, T0=T0, timestep=timestep, minima_traj = "minima.traj")
#
#            num_minima = len([atoms for atoms in list(out_config.to_ConfigSet()) if atoms.info["config_type"].split("_")[-1] == "minima"])
#            print("Number of minima collected", num_minima)
#            count += 1


# perform MinimaHopping on one ASE.atoms object
def _atom_opt_hopping(atom, calculator, Ediff0, T0, minima_threshold, mdmin, parallel, 
                     fmax, timestep, totalsteps, skip_failures, return_all_traj,maxtemp_scale, **opt_kwargs):
    fit_idx = opt_kwargs.pop("fit_idx", 0)
    workdir = os.getcwd()

#    rundir = tempfile.mkdtemp(dir=workdir, prefix='Opt_hopping_', suffix=str(fit_idx))
    if parallel:
        for i in range(parallel):
            rundir = f"{workdir}/parallel/{str(i).zfill(2)}"
            pathlib.Path(f"{workdir}/parallel/{str(i).zfill(2)}").mkdir(parents=True, exist_ok=True)
    else:	
        rundir = f"{workdir}/Opt_hopping_{fit_idx}"
        pathlib.Path(f"{workdir}/Opt_hopping_{fit_idx}").mkdir(parents=True, exist_ok=True)

    os.chdir(rundir)
    atom.calc = calculator
    try:
        opt = MinimaHopping(atom, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold,
                            mdmin=mdmin, fmax=fmax, timestep=timestep, **opt_kwargs)
        print_minhop_parameter(opt)
        opt(totalsteps=totalsteps, maxtemp=maxtemp_scale*T0)
    except Exception as exc:
        # optimization may sometimes fail to converge.
        if skip_failures:
            sys.stderr.write(f'Structure optimization failed with exception \'{exc}\'\n')
            sys.stderr.flush()
            os.chdir(workdir)

            # My implementation
            # If this fails at Optimization step, returns optimization trajectory instead. 
            traj = [] 
            os.chdir(workdir)
#            shutil.rmtree(rundir)	
            return _get_after_explosion(rundir)
#            for opt_traj in Trajectory(glob.glob(f"{rundir}/qn0*.traj")[0]):
#                config_type_append(opt_traj, "hopping_traj")
#                traj.append(opt_traj)
#            print("Optimization failed and trajectories are returned instead", flush=True)
#            return traj

        else:
            raise
    else:
        traj = []
        if return_all_traj:
            traj += _get_MD_trajectory(rundir)

        for hop_traj in Trajectory('minima.traj'):
            config_type_append(hop_traj, 'minima')
            traj.append(hop_traj)
        os.chdir(workdir)
#        shutil.rmtree(rundir)
        return traj


def _run_autopara_wrappable(atoms, calculator, Ediff0=1, T0=1000, minima_threshold=0.5, mdmin=2, parallel=1,
                           fmax=1, timestep=1, totalsteps=10, skip_failures=True, return_all_traj=True, maxtemp_scale=1,
                           autopara_rng_seed=None, autopara_per_item_info=None,
                           **opt_kwargs):
	"""runs a structure optimization

	Parameters
	----------
	atoms: list(Atoms)
		input configs
	calculator: Calculator / (initializer, args, kwargs)
		ASE calculator or routine to call to create calculator
	Ediff0: float, default 1 (eV)
		initial energy acceptance threshold
	T0: float, default 1000 (K)
		initial MD ‘temperature’
	minima_threshold: float, default 0.5 (Å)
		threshold for identical configs
	mdmin: int, default 2
		criteria to stop MD simulation (number of minima)
	fmax: float, default 1 (eV/Å)
		max force for optimizations
	timestep: float, default 1 (fs)
		timestep for MD simulations
	totalsteps: int, default 10
		number of steps
	skip_failures: bool, default True
		just skip optimizations that raise an exception
	opt_kwargs
		keyword arguments for MinimaHopping
	autopara_rng_seed: int, default None
		global seed used to initialize rng so that each operation uses a different but
		deterministic local seed, use a random value if None

	Returns
	-------
		list(Atoms) trajectories
	"""
	calculator = construct_calculator_picklesafe(calculator)
	all_trajs = []

#	if parallel == 1:
	print("serial minima hopping", flush=True)
	for at_i, at in enumerate(atoms_to_list(atoms)):
		if autopara_per_item_info is not None:
			np.random.seed(autopara_per_item_info[at_i]["rng_seed"])

		traj = _atom_opt_hopping(atom=at, calculator=calculator, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold,
								 mdmin=mdmin, fmax=fmax, timestep=timestep, totalsteps=totalsteps,
								 skip_failures=skip_failures, return_all_traj=return_all_traj, **opt_kwargs)
		all_trajs.append(traj)

	return all_trajs

#	elif parallel > 1:
#		print("parallel minima hopping", flush=True)
#		for at_i, at in enumerate(atoms_to_list(atoms)):
#			if autopara_per_item_info is not None:
#				np.random.seed(autopara_per_item_info[at_i]["rng_seed"])
#
#			traj = _atom_opt_hopping(atom=at, calculator=calculator, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold,
#									 mdmin=mdmin, fmax=fmax, timestep=timestep, totalsteps=totalsteps,
#									 skip_failures=skip_failures, return_all_traj=return_all_traj, **opt_kwargs)
#			all_trajs.append(traj)
#
#		return all_trajs
#


# run that operation on ConfigSet, for multiprocessing
def run(*args, **kwargs):
    def_autopara_info = {"num_inputs_per_python_subprocess": 40}

    return autoparallelize(_run_autopara_wrappable, *args,
                           def_autopara_info=def_autopara_info, **kwargs)
autoparallelize_docstring(run, _run_autopara_wrappable, "Atoms")
