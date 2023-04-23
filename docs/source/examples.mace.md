# Parallelize MACE calculator

MACE calculator can be parallelized with the `wfl.calculators.generic.calculate` routine. 

First we define a `ConfigSet` for the inputs and `OutputSpec` to specify how the outputs are handled: 

```
from wfl.configset import ConfigSet, OutputSpec
inputs = ConfigSet("configs.xyz")
outputs = OutputSpec("configs.mace.xyz")
```

Normally, a MACE calculator would be setup like this: 

```
from mace.calculators.mace import MACECalculator 

# change the following as appropriate
mace_model_fname = "my_mace.model"
dtype="float64"

#initialise the calculator
my_mace_calc = MACECalculator(model_path=mace_model_fname, dtype=dtype, device="cpu") 
```

But in Workflow, for `generic.calculate` to parallelize this calculator it needs to be defined as a tuple of `(calc_function, [args], **kwargs)`. In our example, the above code snippet corresponds to 

```
from wfl.calculators import generic
from mace.calculators.mace import MACECalculator 

# change the following as appropriate
mace_model_fname = "my_mace.model"
dtype="float64"

my_mace_calc = (MACECalculator, [], {"model_path":mace_model_fname, "default_dtype":"float64", "device":"cpu"})
```

Now we can evaluate multiple structures in parallel over 8 cores (for example) by exporting (before running the Python script)

```
export WFL_NUM_PYTHON_SUBPROCESSES=8
```

and calling the `generic.calculate`:

```
generic.calculate(
    inputs=inputs, 
    outputs=outputs,
    calculator=my_mace_calc,
    properties = ["energy", "forces"],
    output_prefix="mace_")
```

Since `output_prefix` is set to "mace_" and properties are set to "energy" and "forces", the "structures.mace.xyz" file will have `"mace_energy"` entires in `atoms.info` and `"mace_forces"` entries in `atoms.arrays`. 


## Complete example

1. `export WFL_NUM_PYTHON_SUBPROCESSES=8`

2. run the following script: 

```
from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators import generic
from mace.calculators.mace import MACECalculator 

inputs = ConfigSet("configs.xyz")
outputs = OutputSpec("configs.mace.xyz")


# change the following as appropriate
mace_model_fname = "mace_run-123.model.cpu"

my_mace_calc = (MACECalculator, [], {"model_path":mace_model_fname, "default_dtype":"float64", "device":"cpu"})

generic.calculate(
    inputs=inputs, 
    outputs=outputs,
    calculator=my_mace_calc,
    properties = ["energy", "forces"],
    output_prefix="mace_")

```
