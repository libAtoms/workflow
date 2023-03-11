# Fitting ACE

Workflow provides a wrapper for the [ACE1pack](https://acesuit.github.io/ACE1pack.jl/stable/) package. The function makes use of Workflow's atomic structure handling to fit in with the rest of potential fitting infrastructure and makes use of [ExPyRe](https://libatoms.github.io/ExPyRe/) for submitting just this fitting function as a (remotely) queued cluster job. 

The ace fitting function takes in a parameter dictionary, writes it to (temporary) JSON file and calls ACE1pack's fitting script, `ace_fit.jl`. The executable (e.g. `/path/to/julia $HOME/.julia/packages/ACE1pack/ChRvA/scripts/ace_fit.jl`) is found automatically, unless specified as an argument or via `WFL_ACE_FIT_COMMAND` variable. 

Examples of parameters may be found on [ACE1pack docummentation](https://acesuit.github.io/ACE1pack.jl/stable/command_line/). `wfl.fit.ace.fit()` does some preparation: 

- converts stress to virial 
- sets `energy_key`, etc based on `ref_property_prefix`, i.e. `ace_fit_params["data"]["energy_key"] = f"{ref_property_prefix}energy"`
- parses isolated atom values from the isolated atoms present among the fitting configs
- updates energy/force/virial weights from "energy/force/virial_sigma" `Atoms.info` entries. 

To avoid these modifications, `wfl.fit.ace.run_ace_fit()` can be called directly (which is what `wfl.fit.ace.fit()` calls after the modifications). 
