# Customise Auto-parallelized Functions

```

def optimise(inputs, outputs, calculator, output_prefix, num_inputs_per_python_subprocess=1,
             traj_step_interval=None,num_python_subprocesses=None):
    return autoparallelize(iterable=inputs, outputspec=outputs,
                         calculator=calculator, op=optimise_autopara_wrappable,
                         num_inputs_per_python_subprocess=num_inputs_per_python_subprocess,
                         traj_step_interval=traj_step_interval,
                         output_prefix=output_prefix, num_python_subprocesses=num_python_subprocesses)


def optimise_autopara_wrappable(atoms, calculator, output_prefix, traj_step_interval=None):
    """traj_step_interval: if None, only the last converged config will be
    taken. Otherwise take all that get sampled. + the last

    """

    opt_kwargs = {'logfile': None, 'master': True, 'precon': None,
                  'use_armijo': False, 'steps':500}

    if traj_step_interval is None:
        opt_kwargs["traj_subselect"] = "last_converged"
    if traj_step_interval is not None:
        opt_kwargs['traj_step_interval'] = traj_step_interval

    all_trajs = optimize.run_autopara_wrappable(atoms=atoms, calculator=calculator,
                             keep_symmetry=False, update_config_type=False,
                             results_prefix=output_prefix,
                             fmax=1e-2, **opt_kwargs)

    return all_trajs
```
