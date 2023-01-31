.. _configset:


########################################
Input and output of atomic structures
########################################


``ConfigSet`` and ``OutputSpec`` are python classes defined in ``wfl/configset.py``.


.. code-block:: python

  from wfl.configset import ConfigSet, OutputSpec


``ConfigSet`` can encapsulate one or multiple lists of ``ase.atoms.Atoms`` objects, or reference to stored sets of configuration in files (ABCD databases are currently unsupported). It can function as an iterator over all configs in the input, or iterate over groups of them according to the input definition with the ``ConfigSet().groups()`` method. The ``ConfigSet`` must be initialized with its input configurations, files, or other ``ConfigSet`` objects.

``OutputSpec`` works as the output layer, used for writing results during iterations, but the actual writing is not guaranteed to happen until the operation is closed with ``OutputSpec.close()``. It is possible to map a different output file to each input file, which will result in the outputs corresponding to each input file ending up in a different output file.

Users should consult the simple example in :doc:`first_example`, or the documentation of the two classes at
:meth:`wfl.configset.ConfigSet` and :meth:`wfl.configset.OutputSpec`

==============================
Internals, for developers
==============================
 
For example, to read from two files and write corresponding configs to two other files, use

.. code-block:: python

  configs_in = ConfigSet(['in1.xyz','dir/in2.xyz'])
  s_out = OutputSpec(["out1.xyz",  "out2.xyz"])
  for at in configs_in:
      do_some_operation(at)
      s_out.store(at, at.info.pop("_ConfigSet_loc"))
  s_out.close()
  configs_out = ConfigSet(s_out)


In this case the inputs is a list of files, and the outputs are a mapping between equal number of input and output categories (multiple 1 -> 1).
If the output files were a single string, the mapping woud be many -> 1.

The default behavior for autoparallelized operations is to **skip** an operation if the output files all
exist.  This is made safe(ish) by initially writing the
output to temporary named files, and then renaming them to the actual
requested filename in a ``rename`` operation, so that incomplete files
from an interrupted operation never exist. The ``OutputSpec.all_written()`` 
methods returns true if the ``OutputSpec`` object uses files and appears to 
have already been written (and always returns false for in-memory
storage).

NOTE: the ABCD implementation is not currenly supported, but may be re-added if needed. The previous format is documented here for posterity

.. code-block:: python
  
  s_in = ConfigSet(abcd_conn='mongodb://localhost:27017', input_queries={'input_tag' : 'necessary_input-val'})
  s_out = OutputSpec(abcd_conn='mongodb://localhost:27017', output_abcd=True, set_tags={'output_tag' : 'some unique value'})

In this case the inputs are a dict (single query, AND for each key-value pair) or list of dict for queries (multiple queries, OR of all the dicts), and the output tags are a dict of tags and values to set on writing.  Unless ``output_force=True``, this will refuse  to write if any config already has the output tags set (to ensure that all the configurations written by the loop can be retrieved exactly, for passing to the next stage in the pipeline).  The outputs can be retrieved (outside of the workflow) by

.. code-block:: python 

  abcd.get_atoms(output_tags)

