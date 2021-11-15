""" Tools for GAP xml file operations"""

import xml.etree.ElementTree

import ase.data
import ase.io
import ase.io.extxyz


def extract_e0(filename='GAP.xml', include_zeros=False):
    """Extracts e0 values from a GAP xml file.

    Parameters
    ----------
    filename : path_like
        GAP xml file
    include_zeros : bool
        include zero e0 values, gives a dict complete for all elements

    Returns
    -------
    e0_data : dict
        symbol -> e0_value

    """

    gap = xml.etree.ElementTree.parse(filename).getroot()

    e0_data = dict()

    def _equal_zero(num, tol=1E-5):
        return abs(float(num)) < abs(tol)

    # for one descriptor only now
    for e0_element in gap.findall('GAP_params/GAP_data/e0'):
        sym = ase.data.chemical_symbols[int(e0_element.attrib['Z'])]
        value = float(e0_element.attrib['value'])

        if include_zeros or not _equal_zero(value):
            e0_data[sym] = value

    return e0_data
