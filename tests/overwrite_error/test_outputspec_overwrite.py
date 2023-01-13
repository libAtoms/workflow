import os, json, yaml
from wfl.configset import ConfigSet, OutputSpec
from wfl.descriptors.quippy import calc as calc_descriptor

def get_descriptors(in_file, out_file, gap_params_file, key='desc', **kwargs):
    os.environ['WFL_NUM_PYTHON_SUBPROCESSES'] = "0"
    in_config = ConfigSet(in_file)
    out_config = OutputSpec(files = out_file, overwrite=True)
    gap_params = yaml.safe_load(open(gap_params_file, 'r'))
    gap_params = [i for i in gap_params if 'soap' in i.keys()]
    for param in gap_params:
        param['average'] = True
    calc_descriptor({None: gap_params}, key, per_atom=False, inputs=in_config, outputs=out_config, **kwargs)
    return None

foo = get_descriptors(in_file = "slab.xyz", out_file="slab_desc.xyz", gap_params_file = "desc_dicts.yaml", key = 'desc')
