# Normal Modes of molecules 

Workflow can numerically generate normal modes of a molecule with specified calculator and then simultaneously sample random displacements along multiple normal modes to follow Boltzamnn distribution at a given temperature.  


## Generate

The following script generates normal modes of methane and water with a [xTB](https://xtb-python.readthedocs.io/en/latest/index.html) calculator. The unit normal mode displacements are stored in `Atoms.arrays` and associated frequencies in `Atoms.info`. 

```python
from ase.build import molecule
from xtb.ase.calculator import XTB
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate import normal_modes as nm 

mols = [molecule("CH4"), molecule("H2O")]
configset = ConfigSet(mols)
outputspec = OutputSpec("molecules.normal_modes.xyz")

calc = (XTB, [], {'method':'GFN2-xTB'})
prop_prefix = 'xtb2_'

nm.generate_normal_modes_parallel_hessian(inputs=configset,
                                    outputs=outputspec,
                                    calculator=calc,
                                    prop_prefix=prop_prefix)
```

To generate normal modes via finite differences, each of N atoms are displaced backwards and forwards along each direction leading to 6N calls to the reference calculator. Depending on the number of structures to be processed and speed of the calculator different modes of parallelizing the calculation are appropriate:

- Few structures and slow calculator: generate normal modes in sequence and parallelize the 6N evaluations needed to approximate the Hessian for each molecule. The example above. 
- Many structures and fast calculator: generate normal modes in parallel and evaluate each of the 6N displacements in sequence. Example: 


```python
from ase.build import molecule
from xtb.ase.calculator import XTB
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate import normal_modes as nm 
from wfl.autoparallelize.autoparainfo import AutoparaInfo

mols = [molecule("CH4"), molecule("H2O")]
configset = ConfigSet(mols)
outputspec = OutputSpec("molecules.normal_modes.xyz")

calc = (XTB, [], {'method':'GFN2-xTB'})
prop_prefix = 'xtb2_'

nm.generate_normal_modes_parallel_atoms(inputs=configset,
                                         outputs=outputspec,
                                         calculator=calc,
                                         prop_prefix=prop_prefix,
                                         autopara_info = AutoparaInfo(
                                            num_inputs_per_python_subprocess=1))

```


## Visualize

Normal mode frequencies and displacements can be visualized: 

```python
from ase.io import read, write
from wfl.generate import normal_modes as nm

at = read("molecules.normal_modes.xyz")
water_nm = nm.NormalModes(at, "xtb2_")

# writes trajectories of each normal mode to file
water_nm.view()

# prints frequencies
water_nm.summary()

```


## Sample

Finally, we can generate random displacements of the molecule, along multiple normal modes, that correspond to the Boltzmann distribution at a given temperature. 

```python
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate import normal_modes as nm

inputs = ConfigSet("molecules.normal_modes.xyz")
outputs = OutputSpec("molecules.normal_modes.sample.xyz")

for atoms in inputs:
    at_nm = nm.NormalModes(atoms, "xtb2_")
    sample = at_nm.sample_normal_modes(sample_size=100,
                                        temp=300,  # Kelvin
                                        info_to_keep="default",
                                        arrays_to_keep=None)
    outputs.store(sample)
outputs.close()
```

NB the sampling function clears all but specified `Atoms.info` (apart from `config_type` by default) and `Atoms.arrays` entries to avoid carrying over now incorrect values, for example from previous single point evaluations.
