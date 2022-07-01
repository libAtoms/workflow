# SMILES to .xyz

Needs RDKit installed

## Command line

```
wfl generate-configs smiles -o at.xyz CCCCCC
```

## Python script

Just the "task":
```
at = wfl.generate.smiles.smi_to_atoms("CCCCCC")
```

Parallelised:
```
outputspec = OutputSpec(output_files="compounds.xyz")
smiles = ["CO", "CCCC", "c1ccccc1"]
wfl.generate.smiles.run(outputs=outputspec, smiles=smiles)
```