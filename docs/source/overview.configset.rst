.. _configset:


########################################
Input and output of atomic structures
########################################


``ConfigSet`` and ``OutputSpec`` are python classes defined in ``configset.py``.


.. code-block:: python

  from wfl.configset import ConfigSet, OutputSpec


``ConfigSet`` can encapsulate one or multiple lists of ``ase.atoms.Atoms`` objects, or reference to stored sets of configuration in files or ABCD databases. It can function as an iterator over all configs in the input, or iterate over groups of them according to the input definition with the ``ConfigSet().group_iter()`` method. The ``ConfigSet`` must be initialized with its inputs and indices for them.

``OutputSpec`` works as the output layer, can be used for writing results into it during iterations, but the actual writing is only happening when the operation is closed with ``OutputSpec.end_write()``. Input mapping can be added to output into multiple files, based on the input. This is not fully functional for putting configs into the different outputs in a random order and repeatedly touching one.

For example, to read from two files and write corresponding configs to two other files, use


.. code-block:: python

  s_in = ConfigSet(input_files=['in1.xyz','dir/in2.xyz'])
  s_out = OutputSpec(output_files={"in1.xyz": "out1.xyz", "in2.xyz": "out2.xyz"})
  for at in s_in:
      do_some_operation(at)
      s_out.write(at, from_input_file=s_in.get_current_input_file())
  s_out.end_write()


In this case the inputs is a list of files, and the outputs is either a single file (many -> 1) or a mapping between equal number of input and output categories (multiple 1 -> 1). This will not overwrite unless you also pass ``force=True``.

To read from and write to ABCD database records, you can do


.. code-block:: python

  output_tags = {'output_tag' : 'some unique value'}
  s = ConfigSet(input_abcd='mongodb://localhost:27017' inputs={'input_tag' : 'necessary_input-val'},
              output_abcd='mongodb://localhost:27017', output_tags=output_tags)


In this case the inputs are a dict (single query, AND for each key-value pair) or list of dict for queries (multiple queries, OR of all the dicts), and the output tags are a dict of tags and values to set on writing.  Unless ``output_force=True``, this will refuse  to write if any config already has the output tags set (to ensure  that all the configurations written by the loop can be retrieved exactly, for  passing to the next stage in the pipeline).  The outputs can be retrieved by

.. code-block:: python

  abcd.get_atoms(output_tags)

