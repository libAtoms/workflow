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


#def stratified_random(inputs, outputs, not_enough_minima=False, num=5):
#
#	if outputs.all_written():
#		warnings.warn(f'output {outputs} is done, returning')
#		return outputs.to_ConfigSet()
#
#	minimas = []
#	mdtraj = []
#	new_trainingset = []
#
#	for atoms in inputs:
#		if atoms.info["config_type"].split("_")[-1] == "minima":
#			minimas.append(atoms)
#		else:
#			mdtraj.append(atoms)
#	
#	if len(minimas) > 3:
#		minima_dict = {minima.get_potential_energy(apply_constraint=False) : minima for minima in minimas }
#		print(len(minima_dict))
#		# get one global minimum + two local minima
#		while len(new_trainingset) < 3:
#
#			# get global minimum
#			if len(new_trainingset) == 0:
#				energy = min(minima_dict.keys())
#				if test_short_distance(minima_dict[energy]):
#					print("removed here...")
#					new_trainingset.append(minima_dict.pop(energy))
#				else:
#					print("test not passed")
#					minima_dict.pop(energy)
#
#			# get two local minimum
#			else:
#				energy = random.sample(list(minima_dict), 1)[0]
#				if test_short_distance(minima_dict[energy]):
#					print("removed..")
#					new_trainingset.append(minima_dict.pop(energy))
#				else:
#					print("discarded here...")
#					minima_dict.pop(energy)
#
#	elif len(minimas) <= 3:
#		# In case training set generation failed due to poor quality of MLIP, 
#		# and if the number of training set is not enough, just accept what has been collected as minima. 
#		new_trainingset += minimas
#	else:
#		pass
#
## Get two random structure from MD trajectory
#	random.seed(3)
#	random.shuffle(mdtraj)
#	while len(new_trainingset) < 5:
#		mdstruc = mdtraj.pop()
#		if test_short_distance(mdstruc):
#			new_trainingset.append(mdstruc)
#	
#	for at in new_trainingset:
#		outputs.store(at)
#	outputs.close()
#
#	return outputs.to_ConfigSet()


def stratified_random(inputs, outputs, not_enough_minima=False, num=5):

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
	
	if len(minimas) > 3:
		minimas = sorted(minimas, key=lambda x : -x.get_potential_energy(apply_constraint=False))
	
		print(len(minimas))
		# get one global minimum + two local minima
		while len(new_trainingset) < 3:

			# get global minimum
			if len(new_trainingset) == 0:
				atoms = minimas.pop(0)
				if test_short_distance(atoms):
					print("removed here...")
					new_trainingset.append(atoms)

			# get two local minimum
			else:
				idx = random.sample(range(len(minimas)), 1)[0]
				if test_short_distance(minimas[idx]):
					print("removed..")
					new_trainingset.append(minimas.pop(idx))
				else:
					print("discarded here...")
					minimas.pop(idx)

	elif len(minimas) <= 3:
		# In case training set generation failed due to poor quality of MLIP, 
		# and if the number of training set is not enough, just accept what has been collected as minima. 
		new_trainingset += minimas
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
