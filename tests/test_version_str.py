from wfl.utils.version import get_wfl_version


def test_version_str():
    version_str = get_wfl_version()
    print('version str', version_str)
    assert len(version_str) > 0
