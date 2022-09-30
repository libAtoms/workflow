'''
In this example, we show how "greedy farthest point sampling (FPS)" can be used for selection of "n=10" datapoints from an MD trajectory. This is done in two steps: 1. Assigning  a global descriptor for each configuration in the trajectory followed by 2. A simple call of the function greedy_fps_conf_global in wfl.select.by_descriptor. 
To assign a per-config descriptor we calculate the average SOAP vector for every frame in the MD trajectory. The greedy-FPS algorithm would use them to measure similarities across the datapoints and select 10 unique structures. Overall this requires two input files: the database and the descriptors ("md.xyz" and "gap_params.yaml")
Tip: gap_params.yaml can either be self-written or automatically generated from a template processed by multi-stage gap fit (wfl.fit.gap.multistage)
'''
import os
from wfl.configset import ConfigSet, OutputSpec
from wfl.descriptors.quippy import calc as calc_descriptors
from wfl.select.by_descriptor import greedy_fps_conf_global as get_fps
import yaml

def main(nsamples):
    workdir = os.path.join(os.path.dirname(__file__))
    # Step 1: Assign descriptors to the database
    md        = ConfigSet(os.path.join(workdir, "md.traj"))
    md_desc   = OutputSpec(files=os.path.join(workdir, "md_desc.xyz"))
    
    with open(os.path.join(workdir, 'gap_params.yaml'), 'r') as foo:
        desc_dict = yaml.safe_load(foo)
    desc_dicts = [d for d in desc_dict if 'soap' in d.keys()] # filtering out only SOAP descriptor
    for param in desc_dicts:
        if 'average' not in param.keys():
            param['average']= True # to create global (per-conf) descriptors instead of local (per-atom)
    
    md_desc = calc_descriptors(inputs=md, outputs=md_desc, descs=desc_dicts, key='desc')
    
    # Step 2: Sampling
    fps     = OutputSpec(files=os.path.join(workdir, "out_fps.xyz"))
    get_fps(inputs=md_desc, outputs=fps, num=nsamples, at_descs_info_key='desc', keep_descriptor_info=False)
    return None

if __name__ == '__main__':
    main(nsamples=8)
