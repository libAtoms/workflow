
# Iterative GAP fitting 

The following serves as a basic example of how to fit MLIPs (in this case a GAP for Cu slabs) using the ground functionalities of the wfl package. In theory you need no other previous installations other than wfl, ase, and working versions of QUIP and quippy.

## Table of contents
1. [General workflow and setup](general)
2. [Fitting the initial GAP](fitting-initial)
3. [Preparing the iterative process](preparing-iter)
4. [The iterative Process](iterative-proc)

## General workflow and setup
(general)=

In the ```examples/iterative_gap_fit``` directory you will find the following files:

- batch_gap_fit.py
- EMT_atoms.xyz
- init_md.traj
- multistage_gap_params.json

The iterative fitting workflow is in the file ```batch_gap_fit```. In general, starting from an initial training set of structures with energies and forces, located in ```EMT_atoms.xyz```, and a set of GAP-fit-hyperparameters, located in ```multistage_gap_params.json```, we fit an inital GAP. Then we generate new configs, select a subset of them, and calculate their energies and forces with a *proper* calculator. The new, calculated structures are then added to the initial training set and we fit the second generation GAP. This iterative process is repeated until a maximum iteration.

## Getting started: Parallelization and Imports

Let's take a look at ```batch_gap_fit.py```. In the first lines we define the number of cores over which we parallelize.

```python
import os
os.environ['WFL_NUM_PYTHON_SUBPROCESSES'] = "4"
os.environ['WFL_GAP_FIT_OMP_NUM_THREADS'] = "4"

import wfl.autoparallelize
wfl.autoparallelize.mpipool_support.init(verbose=4)
```
The "WFL_NUM_PYTHON_SUBPROCESSES" being the number of cores you provide the wfl package with, and "WFL_GAP_FIT_OMP_NUM_THREADS" being the number of cores the GAP fit can run in parallel. We initialize the parallelisation by running the [mpipool_support](overview.parallelisation.rst).

Next we import all necessary functions. As you can see, for this example the only non-basic packages are quippy and ase.

```python
import json, os, yaml
import numpy as np

from ase.io import read, write
from ase.calculators.emt import EMT

from pathlib import Path

from quippy.potential import Potential

from wfl.calculators.generic import run as generic_calc
from wfl.descriptors.quippy import from_any_to_Descriptor
from wfl.descriptors.quippy import calc as desc_calc
from wfl.configset import ConfigSet, OutputSpec
from wfl.fit.gap.multistage import prep_params
from wfl.fit.gap.multistage import fit as gap_fit
from wfl.fit.error import calc as ref_calc
from wfl.generate.md import sample as sample_md
from wfl.generate.optimize import optimize 
from wfl.select.by_descriptor import greedy_fps_conf_global
```

## Fitting the initial GAP
(fitting-initial)=

The ```main``` function in ```batch_gap_fit.py``` begins with fitting an initial GAP for Cu structures:

```python
### GAP parameters
gap_params = 'multistage_gap_params.json'
with open(gap_params, 'r') as f:
  gap_params = json.loads(f.read())
Zs = [29]
length_scales = {
  29: {
    "bond_len": [2.6, "NB VASP auto_length_scale"],
    "min_bond_len": [2.2, "NB VASP auto_length_scale"],
    "other links": {},
    "vol_per_atom": [12, "NB VASP auto_length_scale"]
  }
}

training = 'EMT_atoms.xyz'
```
The dictionary of located in ```multistage_gap_params.json``` contains all the necessary GAP hyperparameters for a [multistage fit](wfl.fit.gap.rst). As we are only investigating Cu, the list of unique atomic numbers only has one value, ```Zs=[29]```. To include SOAP heuristics, we add a dictionary of length scales specific to the Cu atom. For more information, we refer to the [universalSOAP package](https://github.com/libAtoms/universalSOAP). To generalize this part of the code for other systems, we suggest using the previously VASP-calculated dictionary located there. The training data is located in the xyz-file ```EMT_atoms.xyz```. This file can be any ase-readable type file, and includes one structure with only one Cu atom, no periodic boundary conditions, and the property ```config_type=isolated_atom```. The energy of this structure being our default Cu energy.

```python
### Initial GAP training
fit_idx = 0
gap_name = f'GAP_{fit_idx}'
GAP_dir = Path('GAP')
GAP_dir.mkdir(exist_ok=True)

if verbose:
  print(f"Fitting original GAP located in {GAP_dir}/{gap_name}.xml",
    flush=True)
get_gap(training, gap_name, Zs, length_scales, gap_params, run_dir=GAP_dir)
```

Next, we create a directory in which we will write all future files resulting from a GAP fit, naming them by the iteration in our process. The function ```get_gap``` represents a helper function that takes training_file, parameters, and output filenames and runs the [multistage gap fit](wfl.fit.rst) function. This function will run locally. If you wish to run this or any other wfl-based function remotely, check out the [Expyre documentation](https://libatoms.github.io/ExPyRe/) and add the remote information via the keyword ```remote_info```.

## Preparing the iterative process
(preparing-iter)=

```python
### MD info
calc = 'md'
MD_dir = Path('MD')
MD_dir.mkdir(exist_ok=True)
md_in_file = 'init_md.traj'
md_configs = read(md_in_file, ':')
md_params = {'steps': 2000, 'dt': 1, 'temperature': 300}

### optimize Info
calc = 'optimize'
optimize_params = {
        "fmax": 0.1,
        "steps": 50,
        "pressure": None,
        "keep_symmetry": False,
        "verbose": True,
    }

n_select = 20
max_count = 5
```

Above are some examples for preparing the [structure generation processes](wfl.generate.rst) ```MD``` and ```optimize```. In this particular file, we define the type of structure generation process with the parameter ```calc```, and set the maximum number of iterations to 5. The variable ```n_select``` determines how many structures get added to the training set each generation.

## The Iterative Process
(iterative-proc)=

```python
while fit_idx  < max_count:
  files = get_file_names(GAP_dir, MD_dir, fit_idx, calc = calc)

  if calc == 'md':
      # Run an MD to create new structures
      run_md(md_configs, files["calc_out"], files["gap"], **md_params)
  elif calc == 'optimize':
      run_optimize(md_configs, files["calc_out"], files["gap"], **optimize_params)

  # Calculate the descriptors for the md output & sample them via fps
  get_descriptors(files["calc_out"], files["desc"], files["gap_params"])
  get_descriptors(training, files["training_desc"], files["gap_params"])
  run_fps(files["desc"], files["fps"], n_select,
      training_desc_file=files["training_desc"]
  )
  run_emt(files["fps"], files["dft"])

  fit_idx += 1
  training_atoms = read(training, ':') + read(files["dft"], ':')
  training = f'{GAP_dir}/training_{fit_idx}.xyz'
  gap_name = f'GAP_{fit_idx}'
  write(training, training_atoms)
  get_gap(training, gap_name, Zs, length_scales, gap_params,
      run_dir=GAP_dir
  )
```

This is where the magic happens. Below we will go through each of the functions in more detail, here we will only discuss the general structure.

After generating structures via MD or optimization, we calculate their global [atomic descriptors](wfl.descriptors.rst) using ```get_descriptors``` for both the generated and the training structures. We use the descriptor hyperparameters from the previous GAP fit. Then we select ```n_select``` structures from the generated structures using [farthest point sampling](wfl.select.rst), adding the training descriptor vectors as reference descriptors.

Next, we calculate the "real" energies and forces on the ```n_select``` structures. For real applications, you would add the *ab initio* code of your choice (see [calculators](wfl.calculators.rst)), for simplicity's sake we use ```ase.emt``` here. These structures then get added to the new training set for iteration ```fit_idx += 1```, and we fit the next generation's GAP.


## Keeping track of training and test errors
```python
val_error   = get_ref_error(files["dft"], files["eval"], files["gap"])

v_f, v_e = 1000 * val_error['forces'], 1000 * val_error['energy']
t_f, t_e = 1000 * train_error['forces'], 1000 * train_error['energy']

if verbose:
  log_dict = {
      "fit_idx": fit_idx,
      "Validation: RMSE_f": v_f, "Training: RMSE_f": t_f,
      "Validation: RMSE_e": v_e, "Training: RMSE_e": t_e
  }

  f_dev = abs(100 * (v_f - t_f)/t_f)
  e_dev = abs(100 * (v_e - t_e)/t_e)

  print(
      f'VALIDATION: RMSE Forces: {v_f:.2f}, RMSE Energy: {v_e:.2f}\n'
      f'TRAINING:   RMSE Forces: {t_f:.2f}, RMSE Energy: {t_e:.2f}\n'
      f'DEVIATIONS: Forces:{f_dev:.2f}%, Energy: {e_dev:.2f}%',
      flush=True
  )
  with open('errors.json', 'a') as f:
      json.dump(log_dict, f)
      f.write('\n')
```
