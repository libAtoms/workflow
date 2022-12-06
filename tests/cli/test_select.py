from click.testing import CliRunner
from ase.io import read, write
from wfl.cli.cli import cli
from wfl.generate import smiles


def test_select_by_lambda(tmp_path):

    fn_out = tmp_path / "ats_out.xyz"
    fn_in = tmp_path / "ats_in.xyz"
    # alkane chanes with 1-10 carbon atoms
    ats_in = [smiles.smi_to_atoms("C"*i) for i in range(1, 11)] 
    write(fn_in, ats_in)
   
    params = [
        '-v',
        'select',
        'lambda',
        f'--outputs {fn_out}',
        f'--inputs {fn_in}',
        # select only structures with even number of carbon atoms
        f'--exec-code "len([sym for sym in list(atoms.symbols) if sym == \'C\']) % 2 == 0"',
    ]

    print(' '.join(params))
    runner = CliRunner()
    result = runner.invoke(cli, " ".join(params))

    ats_out = read(fn_out, ":")
    assert len(ats_out) == 5
