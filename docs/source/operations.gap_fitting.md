# Fitting GAP

`wfl.fig.gap.simple.run_gap_fit()` is a lightweight wrapper for calling `gap_fit` program from python. It makes use of Workflow's atomic structure handling to fit in with the rest of potential fitting infrastructure and makes use of [ExPyRe](https://libatoms.github.io/ExPyRe/) for submitting just this fitting function as a (remotely) queued cluster job. 

The function needs a `fitting_dict` that gets converted in the `gap_fit` command line arguments. The dictionary gets processed so: 

- booleans -> "T"/"F"
- lists ->  { v1 v2 v3 ... }
- {'key1':[v1, v2, v3], 'key2':[v1, v2, v3]} -> '{key1:v1:v2:v3:key2:v1:v2:v3}'
- strings with spaces get enclosed in quotes
- otherwise {key:val} -> key=val
- asserts that mandatory parameters are given
- descriptors are passed in pram_dict['_gap'], which is a list of dictionaries, one dictionary per descriptor
- `atoms_filename` and `at_file` mustn't be present in the dictionary, the file with fitting configs gets created from the input `ConfigSet`. 
- The default executable is `gap_fit`, but it can also be given as an argument or as `WFL_GAP_FIT_COMMAND` environment variable. 

Normally, `OMP_NUM_THREADS` is set to 1, so that only Workflow's parallelization is used and not OpenMP. For fitting GAP, which $isn't parallelizable over atomic structures like other operations are, `OMP_NUM_THREADS` should be set to turn on the OpenMP parallelization. The number of OpenMP threads to to use for fitting GAP, but nothing else, is controlled via  `WFL_GAP_FIT_OMP_NUM_THREADS` environment variable. 

Below is an example dictionary and the corresponding `gap_fit` string (`gap_fit` executable is added later). 

```python
gap_fit_dict = {'default_sigma': [0.01, 0.1, 0.1, 0.0],
                'sparse_seprate_file': False,
                'core_ip_args': 'IP Glue',
                'core_param_file': '/test/path/test/file.xml',
                'config_type_sigma': 'isolated_atom:1e-05:0.0:0.0:0.0:funky_configs:0.1:0.3:0.0:0.0',
                '_gap': [
                    {'soap': True, 'l_max': 6, 'n_max': '12',
                        'cutoff': 3, 'delta': 1,
                        'covariance_type': 'dot_product', 'zeta': 4,
                        'n_sparse': 200, 'sparse_method': 'cur_points'},
                    {'soap': True, 'l_max': 4, 'n_max': 12, 'cutoff': 6,
                        'delta': 1, 'covariance_type': 'dot_product',
                        'zeta': 4, 'n_sparse': 100,
                        'sparse_method': 'cur_points',
                        'atom_gaussian_width': 0.3, 'add_species': False,
                        'n_species': 3, 'Z': 8, 'species_Z': [8, 1, 6]}]}
```

```
gap_fit_string = 'default_sigma={0.01 0.1 0.1 0.0} sparse_seprate_file=F ' \
                'core_ip_args="IP Glue" core_param_file=/test/path/test/file.xml ' \
                'config_type_sigma=isolated_atom:1e-05:0.0:0.0:0.0:' \
                'funky_configs:0.1:0.3:0.0:0.0 ' \
                'gap={ soap=T l_max=6 n_max=12 cutoff=3 delta=1 ' \
                'covariance_type=dot_product zeta=4 n_sparse=200 ' \
                'sparse_method=cur_points : soap=T l_max=4 n_max=12 cutoff=6 delta=1' \
                ' covariance_type=dot_product zeta=4 n_sparse=100 ' \
                'sparse_method=cur_points atom_gaussian_width=0.3 add_species=F ' \
                'n_species=3 Z=8 species_Z={{8 1 6}} }'
```

