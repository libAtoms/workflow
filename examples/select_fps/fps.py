import wfl, os, yaml
from wfl.configset import ConfigSet, OutputSpec
from wfl.descriptors.quippy import calc as calc_descriptors
from wfl.select.by_descriptor import greedy_fps_conf_global as get_fps

work_dir = os.path.join(os.path.dirname(wfl.__file__), "../examples/select_fps") # Location of I/O files

# Step 1: Assign descriptors to the database
md        = ConfigSet(os.path.join(work_dir, "md.traj"))
md_desc   = OutputSpec(files=os.path.join(work_dir, "md_desc.xyz"))

with open(os.path.join(work_dir, 'params.yaml'), 'r') as foo:
    desc_dict = yaml.safe_load(foo)
desc_dicts = [d for d in desc_dict if 'soap' in d.keys()] # filtering out only SOAP descriptor
per_atom = True
for param in desc_dicts:
    if 'average' not in param.keys():
        param['average']= True # to create global (per-conf) descriptors instead of local (per-atom)
        per_atom = False
md_desc = calc_descriptors(inputs=md, outputs=md_desc, descs=desc_dicts, key='desc', per_atom=per_atom)

# Step 2: Sampling
fps      = OutputSpec(files=os.path.join(work_dir, "out_fps.xyz"))
nsamples = 8
get_fps(inputs=md_desc, outputs=fps, num=nsamples, at_descs_info_key='desc', keep_descriptor_info=False)
