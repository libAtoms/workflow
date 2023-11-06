import re
from xml.dom import minidom
from xml.etree import ElementTree

import ase
import ase.data
import numpy as np

from wfl.utils.vol_composition_space import composition_space_Zs


def construct_glue_2b(inputs, energy_info_key, cutoff=None, do_e0=True, filename=None):
    """Construct a glue potential from dimer data, return the isolated atom energies as well

    Parameters
    ----------
    inputs : ConfigSet
    energy_info_key : dict_key
        info key for energy, will not use the calculator's energy directly
    cutoff : float / dict, default None
        cutoff to set for all pairs, can be a float for uniform cutoff, or a dict with keys for pairs in
        'Z1_Z2' format where Z1,Z2 is sorted. Dict with key missing for pair having data defaults to maximum separation
        possible with the given data
    do_e0 : bool, default True
        calculate e0 values, only implemented to be done from homonuclear pairs
    filename : path_like, default None
        if given then xml file is saved to this path

    Returns
    -------
    xml_str : str
        xml param str generated
    e0_by_symbol : dict, None
        dict if do_e0, otherwise None


    """

    def to_key(a, b):
        return "{}_{}".format(*sorted([int(a), int(b)]))

    # find unique atomic numbers
    atomic_numbers = composition_space_Zs(inputs)

    # check if all configs have 2 atoms in them only
    for at in inputs:
        assert len(at) == 2

    # extract per pair energies
    per_pair_data = dict()
    for z1 in atomic_numbers:
        for z2 in atomic_numbers:
            per_pair_data[to_key(z1, z2)] = []

    for at in inputs:
        d = at.get_distance(0, 1)
        energy = at.info[energy_info_key]
        per_pair_data[to_key(*at.get_atomic_numbers())].append([d, energy])

    for key, array in per_pair_data.items():
        if not array:
            # if no dimer data given for this pair
            per_pair_data[key] = None
        else:
            array = np.asarray(array, dtype=float)

            x, unique_index = np.unique(array[:, 0], return_index=True)
            energy_arr = array[unique_index, 1]

            # find cutoff for this pair
            if isinstance(cutoff, (float, int)):
                this_cutoff = float(cutoff)
            elif isinstance(cutoff, dict) and key in cutoff.keys():
                this_cutoff = dict(key)
            else:
                this_cutoff = np.max(x)

            if this_cutoff not in x:
                # insert the next higher or the last lower separation's energy
                if np.any(x > this_cutoff):
                    insert_ener = energy_arr[x > this_cutoff][np.argmin(x[x > this_cutoff])]
                else:
                    insert_ener = energy_arr[x < this_cutoff][np.argmax(x[x < this_cutoff])]
                array = np.vstack([array, [this_cutoff, insert_ener]])

            # only keep data within cutoff
            array = array[array[:, 0] <= this_cutoff, :]
            # sort by distance
            array = array[np.argsort(array[:, 0]), :]

            # save the new array
            per_pair_data[key] = array

    # extract e0
    e0_by_symbol = None
    if do_e0:
        e0_by_symbol = dict()
        for z in atomic_numbers:
            if per_pair_data[to_key(z, z)] is None:
                raise NotImplementedError(f"Missing per-pair dimer data for {z}-{z} for e0 extraction")

            e0_by_symbol[ase.data.chemical_symbols[z]] = per_pair_data[to_key(z, z)][-1, 1] / 2

    # construct glue.xml
    glue = ElementTree.Element("quip")
    ElementTree.SubElement(glue, "Glue_params", n_types=str(len(atomic_numbers)))

    # index for the glue file -- indexing starts from 1
    element_index = {z: i + 1 for i, z in enumerate(atomic_numbers)}

    # density terms, zeros
    for z, index in element_index.items():
        glue[0].append(_element_type_data(index, z))

    # pairwise terms
    for z1 in atomic_numbers:
        for z2 in atomic_numbers:
            if z2 < z1 or per_pair_data[to_key(z1, z2)] is None:
                # count each pair once only and skip ones missing
                continue
            array = per_pair_data[to_key(z1, z2)]
            pot, _ = _element_pair_type(element_index[z1], element_index[z2], array[:, 0], array[:, 1])
            glue[0].append(pot)

    # create xml str
    xml_str = minidom.parseString(ElementTree.tostring(glue)).toprettyxml(indent=" ")
    xml_str = re.sub(r"<.*xml version.*>", "", xml_str)
    xml_str = re.sub(r"[\s]*\n", "\n", xml_str)

    if filename is not None:
        with open(filename, "w") as file:
            file.write(xml_str)

    return xml_str, e0_by_symbol


def _element_type_data(type_index, z):
    per_type_template = """<per_type_data atomic_num="1" type="1">
            <density num_points="2" density_y1="-1.0" density_yn="0.0">
                <point a="1.0" rho="1.0"/>
                <point a="2.0" rho="2.0"/>
            </density>
            <potential_density num_points="2">
                <point rho="1.0" E="0.0"/>
                <point rho="2.0" E="0.0"/>
            </potential_density>
        </per_type_data>"""
    per_type_data = ElementTree.fromstring(per_type_template)
    per_type_data.attrib['atomic_num'] = str(z)
    per_type_data.attrib['type'] = str(type_index)

    return per_type_data


def _element_pair_type(type1, type2, distances, energies):
    template = """<per_pair_data type1="1" type2="1">
            <potential_pair num_points="0" y1="1.0e50" yn="0.0">
            </potential_pair>
            </per_pair_data>"""
    per_pair_data = ElementTree.fromstring(template)

    # check shapes and set number of points
    assert len(distances) == len(energies)
    per_pair_data[0].attrib['num_points'] = str(len(distances))

    # set types
    per_pair_data.attrib['type1'] = str(type1)
    per_pair_data.attrib['type2'] = str(type2)

    # shift the whole curve to have tail at zero
    shift = energies[np.argmax(distances)]

    # add the points
    for r, e in zip(distances, energies):
        per_pair_data[0].append(ElementTree.Element("point", r=str(r), E=str(e - shift)))

    return per_pair_data, shift
