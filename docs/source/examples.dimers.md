# Generating Dimer Structures 

For the generation of machine-learned interatomic potentials dimer curves represent a source of information frequently included in a training set. 
In Workflow the generation of corresponding structures can be performed with the `wfl.generate.atoms_and_dimers.prepare` routine.

The example below illustrates its application to a system containing hydrogen and oxygen atoms. 
At first, we define an `OutputSpec` that will be used to handle the output, i.e. the structural data we are going to generate.
For the generation itself the `prepare()` function is executed where the `atomic_numbers` arguments specifies which combinations 
of species to consider (here all combinations between H and O, i.e. H-H, H-O and O-O). The `bond_lengths` argument allows us to specify 
a characteristic length that is used to sample a range of dimer distances. While isolated atoms are another source of information frequently
added to a training set, in this example we are interested in dimers only and, thus, set `do_isolated_atoms=False`.


```
from wfl.configset import OutputSpec
from wfl.generate.atoms_and_dimers import prepare

dimer = OutputSpec(files='dimers.xyz')
prepare(outputs=dimer, atomic_numbers=[1, 8], bond_lengths={1: 0.74, 8:1.21},
        do_isolated_atoms=False)
```

With Workflow the generation of dimer structures can be as simple as shown in the example. However, additional arguments can be passed 
to the function for more tailored outputs---for instance by adjusting the range of dimer distances to be sampled.

