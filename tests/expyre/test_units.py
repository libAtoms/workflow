import pytest
pytestmark = pytest.mark.remote

from wfl.expyre.units import time_to_HMS, time_to_sec, mem_to_kB


def test_time_to_sec():
    assert time_to_sec(5) == 5
    assert time_to_sec(None) == None

    assert time_to_sec('5') == 5
    assert time_to_sec('10:10') == 10 * 60 + 10
    assert time_to_sec('2:10:10') == 2 * 3600 + 10 * 60 + 10
    assert time_to_sec('3-2:10:10') == 3 * 3600 * 24 + 2 * 3600 + 10 * 60 + 10


def test_time_to_HMS():
    assert time_to_HMS(5) == '0:00:05'
    assert time_to_HMS(1801) == '0:30:01'
    assert time_to_HMS(3700) == '1:01:40'
    assert time_to_HMS(2*24*3600 + 3600 + 10 ) == '49:00:10'


def test_mem_to_kB():
    assert mem_to_kB(5) == 5
    assert mem_to_kB(None) == None

    assert mem_to_kB('5 kB') == 5
    assert mem_to_kB('5kB') == 5
    assert mem_to_kB('5k') == 5

    assert mem_to_kB('5MB') == 5 * 1024

    assert mem_to_kB('5GB') == 5 * 1024**2

    assert mem_to_kB('5TB') == 5 * 1024**3
