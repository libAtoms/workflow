import pytest
from ase.atoms import Atoms
from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators import generic

########################
# test a RuntimeWarning is raised when using the Espresso Calculator directly from ase
from tests.calculators.test_qe import espresso_avail, qe_pseudo
@espresso_avail
def test_wrapped_qe(tmp_path, qe_pseudo):
    from ase.calculators.espresso import Espresso as Espresso_ASE
    from wfl.calculators.espresso import Espresso as Espresso_wrap

    config = Atoms('Si', positions=[[0, 0, 9]], cell=[2, 2, 2], pbc=[True, True, True])

    pspot = qe_pseudo
    kwargs =  dict(
        pseudopotentials=dict(Si=pspot.name),
        pseudo_dir=pspot.parent,
        input_data={"SYSTEM": {"ecutwfc": 40, "input_dft": "LDA",}},
        kpts=(2, 3, 4),
        conv_thr=0.0001,
        workdir=tmp_path,
        tstress=True,
        tprnfor=True
    )

    direct_calc = (Espresso_ASE, [],  kwargs)
    kwargs_generic = dict(inputs=ConfigSet(config), outputs=OutputSpec(), calculator=direct_calc)
    pytest.warns(RuntimeWarning, generic.calculate, **kwargs_generic)
