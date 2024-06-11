
# Calculators in Workflow

In principle, any ASE calculator can be parallelized using Workflow. The parallelization happens at `Atoms` object level. That is, if we wanted to get single-point energies & forces on 16 `Atoms` structures and using 16 cores, all 16 `Atoms` objects would get processed at the same time, each on a single core. 


## Parallelize with `generic.calculate()`

In ASE, we iterate over all `Atoms` objects, initialize a calculator, set it to an `Atoms` object and call it to evaluate energies and forces sequentially. In Workflow, with `generic.calculate` we define a way to initialize a calculator, define where from and to read and write configurations (`ConfigSet` and `OutputSpec`) and set a directive for how many cores to parallelize over. 

The calculator has to be defined as a tuple of `(Calculator, [args], **kwargs)`, for example 

```python
dftb_calc = (
    quippy.potentials.Potential, 
    ["TB DFTB"], 
    {"param_filename": :"tightbind.parms.DFTB.mio-0-1.xm"}
    )
```

Further see [autoparallelization page](overview.parallelisation.rst) and [examples page](examples.index.md).  


## File-based calculators

ASE's calculators that write & read files to & from disk must to be modified if they were to be parallelized via Workflow's `generic` calculator. Specifically, each instance of calculator must execute the calculation in a separate folder so processes running in parallel don't attempt to read and write to the same files. Workflow handles the files, as well as creation and clean-up of temporary directories. 

Currently, ORCA, VASP, QuantumEspresso, CASTEP and FHI-Aims are compatible with the `generic` calculator.

