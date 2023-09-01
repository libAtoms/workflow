import warnings

import numpy as np
import random


def test_short_distance(structure, mindist = 0.1):
    """
    Function for examining if given structure contains too short distance between atoms. 
    mindGist is set to 0.3 Angs which safely avoids any error from FHI-AIMS. 

    Parameters : 
    ------------
    structure : ase Atoms object
        Structure to be examined. 
    mindist : float
        Minimum allowed distance between atoms. 

    Return : 
    Boolean

    """
    distancematrix = structure.get_all_distances()
    distancematrix[np.diag_indices(distancematrix.shape[0])] = 1
    if np.min(distancematrix) < mindist:
        return False
    else:
        return True


def stratified_random(inputs, outputs, num=5, num_minima=3):
	"""
	This function samples geometries from previous global optimization (e.g. minima hopping).
	Some geometries are sampled from local minima and the others are from high-temp MD trajectories.
	If sampled geometry has too short distance, then it's resampled.

    parameters :
    ------------
	inputs: configSet
		atomic configs to select from
	outputs: OutputSpec
		where to write output to
	num: int
		total number to select
	num_minima
		number of local minima configurations to be included in selection	
	
	Return : 
	--------
	selected_configs : ConfigSet
        corresponding to selected configs output
	"""
	if outputs.all_written():
		warnings.warn(f'output {outputs} is done, returning')
		return outputs.to_ConfigSet()

	minimas = []
	mdtraj = []
	new_trainingset = []

	for atoms in inputs:
		if atoms.info["config_type"].split("_")[-1] == "minima":
			minimas.append(atoms)
		else:
			mdtraj.append(atoms)
	
	if len(minimas) > num_minima:
		minimas = sorted(minimas, key=lambda x : -x.get_potential_energy(apply_constraint=False))
	
		# get one global minimum + two local minima
		while len(new_trainingset) < num_minima:

			# get global minimum
			if len(new_trainingset) == 0:
				atoms = minimas.pop(0)
				if test_short_distance(atoms):
					new_trainingset.append(atoms)

			# get two local minimum
			else:
				idx = random.sample(range(len(minimas)), 1)[0]
				if test_short_distance(minimas[idx]):
					new_trainingset.append(minimas.pop(idx))
				else:
					minimas.pop(idx)

	elif len(minimas) <= num_minima:
		# In case training set generation failed due to poor quality of MLIP, 
		# and if the number of training set is not enough, just accept what has been collected as minima. 
		new_trainingset += minimas
	else:
		pass

# Get two random structure from MD trajectory
	random.seed(3)
	random.shuffle(mdtraj)
	while len(new_trainingset) < num:
		mdstruc = mdtraj.pop()
		if test_short_distance(mdstruc):
			new_trainingset.append(mdstruc)
	
	for at in new_trainingset:
		outputs.store(at)
	outputs.close()

	return outputs.to_ConfigSet()
