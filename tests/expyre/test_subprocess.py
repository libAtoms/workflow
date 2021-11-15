import os
from pathlib import Path

import pytest
pytestmark = pytest.mark.remote

from wfl.expyre.subprocess import subprocess_run, subprocess_copy

local_ssh = os.environ.get('EXPYRE_PYTEST_SSH', '/usr/bin/ssh')

def prep(d):
    with open(d / 'file1', 'w') as fout:
        fout.write('\n')
    with open(d / 'file 2', 'w') as fout:
        fout.write('\n')


def test_run_local(tmp_path):
    prep(tmp_path)

    stdout, stderr = subprocess_run(None, ['cd', str(tmp_path), ';', 'ls', 'file1', ';', 'echo', '$USER'])
    assert stdout.splitlines() == ['file1', os.environ['USER']]

    stdout, stderr = subprocess_run(None, ['cd', str(tmp_path), ';', 'ls', 'file 2'])
    assert stdout.splitlines() == ['file 2']


def test_run_remote_same_machine(tmp_path):
    prep(tmp_path)

    stdout, stderr = subprocess_run(os.environ['HOSTNAME'], ['cd', str(tmp_path), ';', 'ls', 'file1', ';', 'echo', '$USER'], remsh_cmd=local_ssh)
    assert stdout.splitlines() == ['file1', os.environ['USER']]

    stdout, stderr = subprocess_run(os.environ['HOSTNAME'], ['cd', str(tmp_path), ';', 'ls', 'file 2'], remsh_cmd=local_ssh)
    assert stdout.splitlines() == ['file 2']


def test_copy_same_machine(tmp_path, change_test_dir):
    prep(tmp_path)

    # make from files in various places under tmp_path
    with open(Path(tmp_path) / 'test_file_1', 'w') as fout:
        fout.write('1\n')
    with open(Path(tmp_path) / 'test_file_2', 'w') as fout:
        fout.write('2\n')
    from_d = (Path(tmp_path) / 'from_dir')
    from_d.mkdir()
    with open(from_d / 'test_file_3', 'w') as fout:
        fout.write('3\n')

    # make to_dir under tmp_path
    to_dir = (Path(tmp_path) / 'to_dir')
    to_dir.mkdir()

    # copy from rel, rel to remote absolute dir
    os.chdir(tmp_path)
    subprocess_copy(['test_file_1', 'test_file_2'], to_dir, to_host=os.environ['HOSTNAME'], remsh_cmd=local_ssh)
    with open(to_dir / 'test_file_1') as fin:
        assert fin.readlines() == [ '1\n' ]
    with open(to_dir / 'test_file_2') as fin:
        assert fin.readlines() == [ '2\n' ]

    # from rel, abs to remote absolute dir
    subprocess_copy(['test_file_1', tmp_path / 'from_dir' / 'test_file_3'], to_dir, to_host=os.environ['HOSTNAME'], remsh_cmd=local_ssh)
    with open(to_dir / 'test_file_1') as fin:
        assert fin.readlines() == [ '1\n' ]
    with open(to_dir / 'test_file_3') as fin:
        assert fin.readlines() == [ '3\n' ]

    # from rel_dir to remote absolute dir
    subprocess_copy('from_dir', to_dir, to_host=os.environ['HOSTNAME'], remsh_cmd=local_ssh)
    with open(to_dir / 'from_dir' / 'test_file_3') as fin:
        assert fin.readlines() == [ '3\n' ]

    # from remote rel to rel dir, violates pytest by using home dir
    assert not (Path.home() / 'pytest_test_file_4').exists()
    with open(Path.home() / 'pytest_test_file_4', 'w') as fout:
        fout.write('4\n')
    subprocess_copy('pytest_test_file_4', to_dir, from_host=os.environ['HOSTNAME'], remsh_cmd=local_ssh)
    with open(to_dir / 'pytest_test_file_4') as fin:
        assert fin.readlines() == [ '4\n' ]
    (Path.home() / 'pytest_test_file_4').unlink()


def test_copy_dry_run(tmp_path):
    prep(tmp_path)

    # test all combinations of relative and absolute, string and path, host and None
    for from_file_1 in ['rel_from_file', '/tmp/abs_from_file']:
        for from_file_2 in [None, 'rel_from_file', '/tmp/abs_from_file']:
            for to_file in ['rel_to_file', '/tmp/abs_to_file']:
                for from_conv in str, Path:
                    for to_conv in str, Path:
                        for host in [None, 'username@host']:
                            # from_file is always a list, later turn len == 1 into scalar
                            if from_file_2 is not None:
                                from_file = [from_file_1, from_file_2]
                            else:
                                from_file = [from_file_1]

                            # do tests with from_host=host
                            from_file_out = []
                            for f in from_file:
                                if host is None and not f.startswith('/'):
                                    # make relative paths on localhost absolute with home
                                    f_out = str(Path.home() / f)
                                else:
                                    f_out = str(f)

                                if host is not None:
                                    f_out = host + ':' + f_out

                                from_file_out.append(f_out)

                            to_file_out = str(to_file)
                            from_file_conv = [from_conv(f) for f in from_file]

                            to_file_conv = to_conv(to_file)

                            if len(from_file_conv) == 1:
                                from_file_conv = from_file_conv[0]

                            args = subprocess_copy(from_file_conv, to_file_conv, from_host=host, dry_run=True)
                            # this depends on args being ['bash', '-lc', 'rsync .....']
                            args = args[0][2].split()

                            assert args[args.index('-a') + 1:] == from_file_out + [to_file_out]

                            # do tests with to_host=host
                            if host is None and not to_file.startswith('/'):
                                # make relative paths on localhost absolute with home
                                to_file_out = str(Path.home() / to_file)
                            else:
                                to_file_out = str(to_file)

                            if host is not None:
                                to_file_out = host + ':' + to_file_out

                            from_file_out = [str(f) for f in from_file]
                            from_file_conv = [from_conv(f) for f in from_file]

                            if len(from_file_conv) == 1:
                                from_file_conv = from_file_conv[0]

                            to_file_conv = to_conv(to_file)
                            args = subprocess_copy(from_file_conv, to_file_conv, to_host=host, dry_run=True)
                            # this depends on args being ['bash', '-lc', 'rsync .....']
                            args = args[0][2].split()

                            assert args[args.index('-a') + 1:] == from_file_out + [to_file_out]
