
# Index

Examples often show examples of more than one thing. Below is a list of common operations with a link to the examples that implement each one. Nearly all examples use `ConfigSet` and `OutputSpec`. [GAP-RSS](workflows.rss.rst) uses most of Workflow's functionalities. 


## Evaluate structures with a calculator
 
- ORCA: [ORCA via python script](examples.orca_python.md)
- EMT: 
    - [First example](first_example.md)
    - [Iterative GAP fitting ](examples.mlip_fitting.md)
- XTB: 
    - [Normal Modes of molecules](examples.normal_modes.md)
    - [Molecular Dynamics](examples.md.md)
    - [GAP fit workflow with many wfl use-case examples ](examples.daisy_chain_mlip_fitting.ipynb)
- VASP: [Functions as independently queued jobs](overview.queued.md) 
- FHI-Aims: [FHI-Aims Calculator](examples.fhiaims_calculator.ipynb) 


## Generate structures

- [Generate Dimer Structures](examples.dimers.ipynb) 
- [Random Structures via buildcell](examples.buildcell.ipynb)
- From SMILES string: 
    - Short example in [SMILES to `Atoms`](examples.smiles.md)
    - As part of [GAP fit workflow with many wfl use-case examples ](examples.daisy_chain_mlip_fitting.ipynb)
- Geometry optimisation: [Iterative GAP fitting ](examples.mlip_fitting.md)
- Sample molecular normal modes: [Normal Modes (non-periodic)](examples.normal_modes.md)


### Run MD

- Sample configs for training: 
    - [Iterative GAP fitting ](examples.mlip_fitting.md)
    - [GAP fit workflow with many wfl use-case examples ](examples.daisy_chain_mlip_fitting.ipynb)
- Run ACE MD: [MD](examples.md.md)


## Remote execution

- Overview: [Functions as independently queued jobs](overview.queued.md)
- [ORCA via python script](examples.orca_python.md)
- Run ACE MD: [MD](examples.md.md)
- [GAP fit workflow with many wfl use-case examples ](examples.daisy_chain_mlip_fitting.ipynb)


## Get descriptors 

- Global SOAP: 
    - [Iterative GAP fitting ](examples.mlip_fitting.md)
    - [GAP fit workflow with many wfl use-case examples ](examples.daisy_chain_mlip_fitting.ipynb)


## Sampling structures

- Furthest point sampling: [Sampling of Structures](examples.select_fps.ipynb)
- CUR: [GAP fit workflow with many wfl use-case examples ](examples.daisy_chain_mlip_fitting.ipynb)
- With a boolean function: [GAP fit workflow with many wfl use-case examples ](examples.daisy_chain_mlip_fitting.ipynb)



## Fit a potential

- Multistage GAP fit: [Iterative GAP fitting ](examples.mlip_fitting.md)
- Simple GAP fit: [GAP fit workflow with many wfl use-case examples ](examples.daisy_chain_mlip_fitting.ipynb)


## Iterative training

- An example of GAP fitting scheme: [Iterative GAP fitting ](examples.mlip_fitting.md)


## Command line 

- Generate structures from SMILES: [SMILES to `Atoms`](examples.smiles.md) 


## Miscellaneous

- Calculate errors: 
    - [Iterative GAP fitting ](examples.mlip_fitting.md)
    - [GAP fit workflow with many wfl use-case examples ](examples.daisy_chain_mlip_fitting.ipynb)
- Plot predicted vs reference correlation plots: [GAP fit workflow with many wfl use-case examples ](examples.daisy_chain_mlip_fitting.ipynb)
- Parallelize your own function: [Overview of](overview.parallelisation.rst)
- Calculate normal modes of a molecule: [Normal Modes(non-periodic)](examples.normal_modes.md)
- Post-process each ORCA calculation on-the-fly: [ORCA via python script](examples.orca_python.md)
