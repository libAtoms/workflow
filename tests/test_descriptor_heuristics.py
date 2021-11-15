import os
import json

from wfl.utils.round_sig_figs import round_sig_figs
from wfl.descriptor_heuristics import descriptors_from_length_scales

def test_descriptors_from_length_scales():
    Zs = [6, 14]
    desc_templ = [ { 'neighbors_Nb': True, 'order': 2, 'cutoff': '_EVAL_ {BOND_LEN_Z1_Z2}*3', 
                     'Z': [ '_EVAL_ {Z1}', '_EVAL_ {Z2}'], 
                     'add_species': 'manual_Z_pair' },
                   { 'soap': True, 'n_max': 3, 'cutoff': '_EVAL_ {BOND_LEN_Z_MAX}*2', 'atom_sigma': '_EVAL_ {BOND_LEN_Z}/3',
                     'Z': '_EVAL_ {Zcenter}', 'n_species': '_EVAL_ {nZ}', 'species_Z': '_EVAL_ {Zs}',
                     'add_species': 'manual_Zcenter' },
                   { 'soap': True, 'n_max': 3, 'cutoff': '_EVAL_ {R_CUT}', 'atom_sigma': '_EVAL_ {ATOM_SIGMA}',
                     'Z': '_EVAL_ {Zcenter}', 'n_species': '_EVAL_ {nZ}', 'species_Z': '_EVAL_ {Zs}',
                     'add_species': 'manual_universal_SOAP' }
                 ]

    length_scales = {6:  {'bond_len': [10, 'arbitrary'] },
                     14: {'bond_len': [20, 'arbitrary'] } }

    SOAP_hypers = {6:  [{'cutoff': 3.03, 'cutoff_transition_width': 0.35, 'atom_gaussian_width': 0.5},
                        {'cutoff': 4.03, 'cutoff_transition_width': 0.5, 'atom_gaussian_width': 1.0} ],
                   14: [{'cutoff': 5.03, 'cutoff_transition_width': 0.75, 'atom_gaussian_width': 1.5}] }

    descs, desc_Zs = descriptors_from_length_scales(desc_templ, Zs, length_scales, SOAP_hypers)

    actual_by_key = {}
    for Zc, desc in zip(desc_Zs, descs):
        try:
            Zc_key = tuple(sorted(Zc))
        except:
            Zc_key = Zc
        if Zc_key not in actual_by_key:
            actual_by_key[Zc_key] = []
        actual_by_key[Zc_key].append(desc)

    # MAKE REFERENCE
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'descriptor_heuristics.json')) as fin:
        ref_data = json.load(fin)
    desc_Zs_ref = ref_data['desc_Zs']
    descs_ref = ref_data['descs']

    ref_by_key = {}
    for Zc, desc in zip(desc_Zs_ref, descs_ref):
        try:
            Zc_key = tuple(sorted(Zc))
        except:
            Zc_key = Zc
        if Zc_key not in ref_by_key:
            ref_by_key[Zc_key] = []
        ref_by_key[Zc_key].append(desc)

    # import pprint
    # print('ACTUAL')
    # pprint.pprint(actual_by_key)
    # print('REF')
    # pprint.pprint(ref_by_key)

    assert set(actual_by_key.keys()) == set(ref_by_key.keys())
    for key in actual_by_key.keys():
        print('check key', key)
        found = False
        for act_entry in actual_by_key[key]:
            print('look for', act_entry)
            for ref_entry_i, ref_entry in enumerate(ref_by_key[key]):
                if act_entry == ref_entry:
                    found=True
                    break
            print('found', found)
            assert found
            del ref_by_key[key][ref_entry_i]
        print('final len', len(ref_by_key[key]))
        assert len(ref_by_key[key]) == 0
