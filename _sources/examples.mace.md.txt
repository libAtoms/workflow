# Parallelize MACE calculator

MACE calculator can be parallelized with the `wfl.calculators.generic.run` routine. 

First we define a `ConfigSet` for the inputs and `OutputSpec` to specify how the outputs are handled: 

```
from wfl.configset import ConfigSet, OuptutSpec
inputs = ConfigSet(input_files="structures.xyz")
outputs = OutputSpec(output_files="structures.mace.xyz")
```

Normally, a MACE calculator would be setup like this: 

```
from mace.calculators.mace import MACECalculator 

# change the following as appropriate
mace_model_fname = "my_mace.model"
r_max = 5.0, 
at_num = [1, 6], 
dtype="float64"

#initialise the calculator
my_mace_calc = MACECalculator(mace_model_fname, r_max=r_max, at_num=at_num, dtype=dtype, device="cpu") 
```

But in Workflow, for `generic.run` to parallelize this calculator it needs to be defined as a tuple of `(calc_function, [args], **kwargs)`. In our example, instead of the above code snippet, that corresponds to 

```
from mace.calculators.mace import MACECalculator 

# change the following as appropriate
mace_model_fname = "my_mace.model"
r_max = 5.0, 
at_num = [1, 6], 
dtype="float64"

my_mace_calc = (MaceCalculator, [mace_model_fname], {"r_max":r_max, "at_num": at_num, "dtype":dtype, "device":"cpu})
```

Now we can evaluate multiple structures in parallel over 8 cores (for example) by exporting (before running the Python script)

```
export WFL_NUM_PYTHON_SUBPROCESSES=8
```

and calling the `generic.run`:

```
generic.run(
    inputs=inputs, 
    outputs=outputs,
    calculator=my_mace_calc,
    properties = ["energy", "forces"],
    output_prefix="mace_")
```

Since `output_prefix` is set to "mace_" and properties are set to "energy" and "forces", the "structures.mace.xyz" file will have `"mace_energy"` entires in `atoms.info` and `"mace_forces"` entries in `atoms.arrays`. 


The complete process 

1. `export WFL_NUM_PYTHON_SUBPROCESSES=8`

2. run the following script: 

```
from wfl.configset import ConfigSet, OuptutSpec

from mace.calculators.mace import MACECalculator 

inputs = ConfigSet(input_files="structures.xyz")
outputs = OutputSpec(output_files="structures.mace.xyz")


# change the following as appropriate
mace_model_fname = "my_mace.model"
r_max = 5.0, 
at_num = [1, 6], 
dtype="float64"

my_mace_calc = (MaceCalculator, [mace_model_fname], {"r_max":r_max, "at_num": at_num, "dtype":dtype, "device":"cpu})

generic.run(
    inputs=inputs, 
    outputs=outputs,
    calculator=my_mace_calc,
    properties = ["energy", "forces"],
    output_prefix="mace_")

```