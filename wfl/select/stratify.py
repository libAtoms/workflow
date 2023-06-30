import warnings

import numpy as np
import random


def test_short_distance(structure, mindist = 0.3):
    """
    Function for examining if given structure contains too short distance between atoms. 
    mindist is set to 0.3 Angs which safely avoids any error from FHI-AIMS. 

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


def stratified_random(inputs, outputs, num=5):

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
	
	if len(minimas) != 0:
		minima_dict = {minima.get_potential_energy(apply_constraint=False) : minima for minima in minimas }

		# get one global minimum + two local minima
		while len(new_trainingset) < 3:

			# get global minimum
			if len(new_trainingset) == 0:
				energy = min(minima_dict.keys())
				if test_short_distance(minima_dict[energy]):
					new_trainingset.append(minima_dict.pop(energy))
				else:
					minima_dict.pop(energy)

			# get two local minimum
			else:
				energy = random.sample(list(minima_dict), 1)[0]
				if test_short_distance(minima_dict[energy]):
					new_trainingset.append(minima_dict.pop(energy))
				else:
					minima_dict.pop(energy)
	else:
		pass

# Get two random structure from MD trajectory
	random.seed(3)
	random.shuffle(mdtraj)
	while len(new_trainingset) < 5:
		mdstruc = mdtraj.pop()
		if test_short_distance(mdstruc):
			new_trainingset.append(mdstruc)
	
	for at in new_trainingset:
		outputs.store(at)
	outputs.close()

	return outputs.to_ConfigSet()
