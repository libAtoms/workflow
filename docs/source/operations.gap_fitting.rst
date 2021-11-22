.. _gap_fitting:

##################################################
Multistage GAP fitting 
##################################################


Fit a GAP with multiple descriptor in multiple stages, to calculate the appropriate delta (energy scale) at each stage

**************************************************
Procedure
**************************************************


* calculate spread of energy in reference data, use for delta in stage 0

  * delta =~ energy spread / n_descriptors_per_atom

* modify each config in fitting database, e.g. setting per-config ``energy_sigma``, ``force_sigma``, using a callback in a python module 
* fit GAP(stage=0), naming it with ``{label}.stage_{stage}.xml`` 
* repeat for stage=1..

  * calculate fitting error for GAP(stage-1)
  * use fitting error (residual) to estimate delta for this stage's descriptors

    * delta =~ fitting error / n_descriptors_per_atom

  * modify each config in fitting database as in stage 0
  * fit GAP(stage)

* rename final GAP file to ``{label}.xml``


**************************************************
Configuration file
**************************************************


The iterative fitting procedure is controlled by a JSON file that describes the descriptors and fitting params for each stage, and some global fitting params.

* The top level dict contains two keys, ``stages`` and ``gap_params``

  * ``stages`` : list of dicts, one for each stage. 

    * Each stage contains a dict with keys ``error_scale_factor`` and ``descriptors``

      * ``error_scale_factor``: factor to apply to all fitting sigmas when doing the fit for that stage (e.g. to reduce desired accuracy of stage 0 2-body only fit), defaults to 1.0 if missing.  Passed to database modifying module callback mentioned above.
      * ``descriptors``: list of dicts describing each descriptor and its fitting params.

        * Each list item contains a dict with ``desc_str``, ``fit_str``, and ``count_cutoff``

          * ``desc_str``: string used to create QUIP ``Descriptor``
          * ``fit_str``: string used to create the rest of the ``gap_fit`` input arguments, e.g. ``n_sparse``, ``covariance_type``, etc.
          * ``count_cutoff``, optional: cutoff to be used when counting descriptors per atom, used for counting only close neighbors for 2-body descriptors, even if they are actually much longer ranged.

  * ``gap_params`` : global ``gap_fit`` params, e.g. ``default_sigma``, ``sparse_jitter``, ``output_separate_file``

**************************************************
Creating configuration file
**************************************************


``wfl.fit.gap_multistage`` has a function ``prep_input`` which takes a _template_ configuration file and creates descriptors for each species using universal SOAP hyperparameters.  The template file format is similar to the configuration file described above.  The main exceptions are

* in addition to the ``error_scale_factor`` and ``descriptors`` keys there is another, ``add_species``
* All strings (mainly intended for ``desc_str``) have length scales specified in terms of ``${REPL_EXPR}``, which will be replaced by the mathematical expression, substituting particular strings (e.g. ``R_CUT``, ``BOND_LEN_Z``) as described in the docstring for ``wfl.descriptor_heuristics.dup_descs_for_species()``.  Replacements will remain strings, except strings that begin 
  with ``_F_``, which will be replaced with the evaluate floating point value.


``add_species`` can be any value understood by ``wfl.descriptor_heuristics.dup_descs_for_species()``, in particular

* ``manual_Z1_Z2`` for 2-body descriptors with length scale set differently for each Z1-Z2 pair
* ``manual_universal_SOAP`` for universal SOAPs that will be created, 2 or more, for each center Z, using heuristics
