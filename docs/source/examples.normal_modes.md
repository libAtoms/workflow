# Normal Modes (non-periodic)

## Generate

```
@click.command('xtb-normal-modes')
@click.argument('input-fname')
@click.option('-o', '--output-fname')
@click.option('--parallel-hessian', "parallel_hessian", flag_value=True,
              default=True)
@click.option('--parallel-atoms', "parallel_hessian", flag_value=False)
def xtb_normal_modes(input_fname, output_fname, parallel_hessian):

    from xtb.ase.calculator import XTB

    ConfigSet = ConfigSet(input_fname)
    OutputSpec = OutputSpec(output_fname)

    calc = (XTB, [], {'method':'GFN2-xTB'})

    prop_prefix = 'xtb2_'

    if parallel_hessian:
        vib.generate_normal_modes_parallel_hessian(inputs=ConfigSet,
                                          outputs=OutputSpec,
                                          calculator=calc,
                                          prop_prefix=prop_prefix)
    else:
        vib.generate_normal_modes_parallel_atoms(inputs=ConfigSet,
                                                 outputs=OutputSpec,
                                                 calculator=calc,
                                                 prop_prefix=prop_prefix,
                                                 num_inputs_per_python_subprocess=1)
```


## Sample

```
def sample(inputs, outputs, temp, sample_size, prop_prefix,
                        info_to_keep=None, arrays_to_keep=None):
    """Multiple times displace along normal modes for all atoms in input

    Parameters
    ----------

    inputs: Atoms / list(Atoms) / ConfigSet
        Structures with normal mode information (eigenvalues &
        eigenvectors)
    outputs: OutputSpec
    temp: float
        Temperature for normal mode displacements
    sample_size: int
        How many perturbed structures per input structure to return
    prop_prefix: str / None
        prefix for normal_mode_frequencies and normal_mode_displacements
        stored in atoms.info/arrays
    info_to_keep: str, default "config_type"
        string of Atoms.info.keys() to keep
    arrays_to_keep: str, default None
        string of Atoms.arrays.keys() entries to keep

    Returns
    -------
    """

    if isinstance(inputs, Atoms):
        inputs = [inputs]

    for atoms in inputs:
        at_vib = Vibrations(atoms, prop_prefix)
        sample = at_vib.sample_normal_modes(sample_size=sample_size,
                                            temp=temp, 
                                            info_to_keep=info_to_keep,
                                            arrays_to_keep=arrays_to_keep)
        outputs.store(sample)

    outputs.close()
    return ConfigSet(outputs)
```

Elena: I have [this script](https://github.com/gelzinyte/scripties/blob/main/util/normal_modes.py) that randomly displaces atoms along different normal modes and down-weights the very shallow noisy ones. Is it worth merging into workflow and/or describing here? 
