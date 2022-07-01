
# Calculators in Workflow

In principle, any ASE calculator can be parallelized using Workflow. The parallelization happens at `Atoms` object level. That is, if we wanted to get single-point energies & forces on 16 `Atoms` structures and using 16 cores, all 16 `Atoms` object would get processed at the same time, each on a single core. 


## Parallelize with `generic.run()`

In ASE, we iterate over all `Atoms` objects, initialize a calculator, set it to an `Atoms` object and call it to evaluate energies and forces sequentially. In Workflow, with `generic.run` we define a way to initialise a calculator, define where from and to read and write configurations (`ConfigSet` and `OutputSpec`, see {TODO: add page links}) and set a directive for how many cores to paralellize over. 

The calculator has to be defined as a tuple of `(Calculator, [args], **kwargs)`, for example 

```python
dftb_calc = (
    quippy.potentials.Potential, 
    ["TB DFTB"], 
    {"param_filename": :"tightbind.parms.DFTB.mio-0-1.xm"}
    )
```

See [autoparallelization page](overview.parallelisation.rst) for further explanation, [MACE example](examples.mace.md) and [wfl.calculators.generic.run](??) for more details.  


## File-based calculators

ASE's calculators that write & read files to & from disk (subclasses of `FileIOCalculator` (TODO check)) need to be slightly modified if they were to be parallelized via Workflow's `generic` calculator. Specifically, each instance of calculator must execute the calculation in a separate folder. Workflow handles the files, as well as creation and clean-up of temporary directories. 

CASTEP ([example](link), [docs](link)), VASP ([example](link), [docs](link)) and QuantumESPRESSO ([example](link), [docs](link)) are accessed via `calculators.dft.evaluate_dft()` ([docs](link)). ORCA calculator ([example](examples.orca.md), [docs](link)) can be parallelized with the `generic` calculator. 


## Special calculators

Finally, there is a non-conventional "Basin Hopping" calculator. 
`BasinHoppingORCA()` ([Example](link), [docs](link), [publication](link)) runs multiple single point evaluations, perturbing the initial guess of the wavefunction each time. It returns the results corresponding to the global minima and lowest-energy solution.  

