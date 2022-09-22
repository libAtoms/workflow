'''
In this example, we show how "greedy farthest point sampling (FPS)" can be used for selection of "n=10" datapoints from an MD trajectory. This is done in two steps: 1. Assigning  a global descriptor for each configuration in the trajectory followed by 2. A simple call of the function greedy_fps_conf_global in wfl.select.by_descriptor. 
To assign a per-config descriptor we calculate the average SOAP vector for every frame in the MD trajectory. The greedy-FPS algorithm would use them to measure similarities across the datapoints and select 10 unique structures. Overall this requires two input files: the database and the descriptors ("md.xyz" and "gap_params.yaml")
Tip: gap_params.yaml can either be self-written or automatically generated from a template processed by multi-stage gap fit (wfl.fit.gap.multistage)
'''

from wfl.configset import ConfigSet, OutputSpec
from wfl.descriptors.quippy import calc as calc_descriptors
from wfl.select.by_descriptor import greedy_fps_conf_global as get_fps
import yaml

# Step 1: Assign descriptors to the database
md        = ConfigSet(input_files="md.xyz")
md_desc   = OutputSpec(output_files="md_desc.xyz")

desc_dict = yaml.safe_load(open('gap_params.yaml', 'r'))
desc_dicts = [d for d in desc_dict if 'soap' in d.keys()] # filtering out only SOAP descriptor
for param in desc_dicts:
    if 'average' not in param.keys():
        param['average']= True # to create global (per-conf) descriptors instead of local (per-atom)

calc_descriptors(inputs=md, outputs=md_desc, descs=desc_dicts, key='desc')

# Step 2: Sampling
md_desc = ConfigSet(input_files="md_desc.xyz")
fps     = OutputSpec(output_files="fps.xyz")

get_fps(inputs=md_desc, outputs=fps, num=10, at_descs_info_key='desc', keep_descriptor_info=False)
