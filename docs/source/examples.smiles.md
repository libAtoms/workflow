# SMILES to `Atoms` 

Conversion of [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) to ASE's `Atoms` is done via [RDKit](http://rdkit.org/). To install: 

```
conda install -c conda-forge rdkit
```

## Command line

```
wfl generate-configs smiles -o configs.xyz CCCCCC CC c1ccccc1
```

## Python script

Single operation:

```python
from wfl.generate import smiles
atoms = smiles.smi_to_atoms("CCCCCC")
```

With Workflow's parallelization:

```python
from wfl.configset import ConfigSet
from wfl.generate import smiles

outputspec = OutputSpec("compounds.xyz")
smiles = ["CO", "CCCC", "c1ccccc1"]
smiles.smiles(smiles, outputs=outputspec)
```

NB `smiles` has to be given as the first argument. 
