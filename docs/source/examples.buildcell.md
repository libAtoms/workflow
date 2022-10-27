# Random Structures via buildcell

In Workflow random structures can be generated via the `wfl.generate.buildcell.run()` routine. 
It's functionality builds on [AIRSS](https://airss-docs.github.io/technical-reference/buildcell-manual)’s `buildcell` to constructs sensible random structures.

The example below illustrates its application for the random generation of aluminum unit cells.
Here, we are aiming for a total of 20 structures and, thus, set `inputs` to an iterable of that length.
Next, we define an `OutputSpec` to handle the output structures that will be generated.
In order to have a proper `buildcell_input` available, we are using the `wfl.generate.buildcell.create_input()` routine in this example
where we pass arguments that characterise the systems we are aiming to generate.
Finally, we set the `buildcell_cmd` appropriately to the `buildcell` executable we use on our machine and run the script
to obtain the desired number of random Al-based unit cells.

```
from wfl.generate.buildcell import create_input, run
from wfl.configset import OutputSpec


inputs = range(20)
outputspec = OutputSpec('buildcell_output.xyz')

input_file = 'buildcell_input.cell'
create_input(z=13, vol_per_atom=10, bond_lengths=2, filename=input_file)
with open(input_file, 'r') as fo:
    buildcell_input = fo.read()

run(inputs=inputs,
    outputs=outputspec,
    buildcell_input=buildcell_input,
    buildcell_cmd='buildcell',
    )
```
