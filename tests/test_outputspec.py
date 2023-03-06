import pytest

from wfl.configset import ConfigSet, OutputSpec
from ase.atoms import Atoms
import ase.io

@pytest.fixture
def ats():
    return [Atoms(numbers=[Z]) for Z in range(1, 1+10)]

def gather_numbers(items):
    items_out = []
    for item in items:
        if isinstance(item, Atoms):
            items_out.append(item.numbers.tolist())
        elif isinstance(item, ConfigSet):
            items_out.append(gather_numbers(item.groups()))
        else:
            items_out.append(gather_numbers(item))

    return items_out

def check_ConfigSet(cs, locs=None, numbers=None):
    # print("ConfigSet", cs)
    # for at in cs:
        # print(f"{at.numbers} info['_ConfigSet_loc'] '{at.info.get('_ConfigSet_loc')}' cur_loc '{cs.cur_loc}'")

    if locs is not None:
        ## print("BOB locs", locs)
        ## print("BOB actual locs", [at.info["_ConfigSet_loc"] for at in cs])
        assert locs == [at.info["_ConfigSet_loc"] for at in cs]

    # print("ConfigSet as tree")
    # print_tree(cs.groups(), "  ")

    if numbers is not None:
        ## print("BOB numbers", numbers)
        ## print("BOB gathered numbers", gather_numbers(cs.groups()))
        assert numbers == gather_numbers(cs.groups())

def test_roundtrip_mem_mem(ats):
    ats_i = [[ats[0:2], ats[2:4]], [ats[4:6]]]
    cs = ConfigSet(ats_i)
    locs = ["/0/0/0", "/0/0/1", "/0/1/0", "/0/1/1", "/1/0/0", "/1/0/1"]
    locs = [l.replace("/", " / ") for l in locs]

    check_ConfigSet(cs, locs, gather_numbers(ats_i))

    os = OutputSpec()
    for at in cs:
        os.store(at, input_CS_loc = at.info.pop("_ConfigSet_loc"))
    os.close()
    cs_o = os.to_ConfigSet()

    check_ConfigSet(cs_o, locs, gather_numbers(ats_i))

def test_roundtrip_mem_mem_no_loc(ats):
    ats_i = ats[0:6]
    cs = ConfigSet(ats_i)
    locs = [f"/{i}" for i in range(len(ats_i))]
    locs = [l.replace("/", " / ") for l in locs]

    check_ConfigSet(cs, locs, gather_numbers(ats_i))

    os = OutputSpec()
    for at in cs:
        os.store(at, input_CS_loc = at.info.pop("_ConfigSet_loc"))
    os.close()
    cs_o = os.to_ConfigSet()

    check_ConfigSet(cs_o, locs, gather_numbers(ats_i))

def test_roundtrip_mem_one_file(tmp_path, ats):
    ats_i = [[ats[0:2], ats[2:4]], [ats[4:7]]]
    cs = ConfigSet(ats_i)
    locs = [l.replace("/", " / ") for l in ["/0/0/0", "/0/0/1", "/0/1/0", "/0/1/1", "/1/0/0", "/1/0/1", "/1/0/2"]]

    check_ConfigSet(cs, locs, gather_numbers(ats_i))

    os = OutputSpec("ats.xyz", file_root=tmp_path)
    for at in cs:
        os.store(at, input_CS_loc = at.info.pop("_ConfigSet_loc"))
    os.close()
    cs_o = os.to_ConfigSet()

    check_ConfigSet(cs_o, locs, gather_numbers(ats_i))

def test_roundtrip_mem_mult_files(tmp_path, ats):
    ats_i = [[ats[0:2], ats[2:4]], [ats[4:7]]]
    cs = ConfigSet(ats_i)
    locs = [l.replace("/", " / ") for l in ["/0/0/0", "/0/0/1", "/0/1/0", "/0/1/1", "/1/0/0", "/1/0/1", "/1/0/2"]]

    check_ConfigSet(cs, locs, gather_numbers(ats_i))

    os = OutputSpec(["ats_0.xyz", "ats_1.xyz"], file_root=tmp_path)
    for at in cs:
        os.store(at, input_CS_loc = at.info.pop("_ConfigSet_loc"))
    os.close()
    cs_o = os.to_ConfigSet()

    check_ConfigSet(cs_o, locs, gather_numbers(ats_i))

def test_without_input_CS_loc(tmp_path, ats):
    os = OutputSpec("ats.xyz", file_root=tmp_path)
    for at in ats:
        os.store(at)
    os.close()

    for at_i, at in enumerate(ase.io.read(tmp_path / "ats.xyz", ":")):
        assert at.info["_ConfigSet_loc"] == ConfigSet._loc_sep + str(at_i)

def test_mixed_input_CS_loc_presence(tmp_path, ats):
    with pytest.raises(RuntimeError):
        os = OutputSpec("ats.xyz", file_root=tmp_path)
        for at_i, at in enumerate(ats):
            if at_i < 3:
                os.store(at, input_CS_loc = ConfigSet._loc_sep + str(at_i))
            else:
                os.store(at)
        os.close()

    with pytest.raises(RuntimeError):
        os = OutputSpec("ats.xyz", file_root=tmp_path)
        for at_i, at in enumerate(ats):
            if at_i > 3:
                os.store(at, input_CS_loc = ConfigSet._loc_sep + str(at_i))
            else:
                os.store(at)
        os.close()

def test_overwrite(tmp_path, ats):
    ase.io.write(tmp_path / "ats.xyz", ats)

    # default overwrite=False
    os = OutputSpec("ats.xyz", file_root=tmp_path)
    # should not fail yet
    with pytest.raises(FileExistsError):
        for at in ats:
            os.store(at)
    os.close()

    # force overwrite=True
    os = OutputSpec("ats.xyz", file_root=tmp_path, overwrite=True)
    for at in ats:
        os.store(at)
    os.close()

def test_write_configset(tmp_path, ats):
    c = ConfigSet(ats)

    os = OutputSpec("ats.xyz", file_root=tmp_path)
    os.write(c)

    for a0, a1 in zip(c, os.to_ConfigSet()):
        assert a0 == a1

def test_tags(tmp_path, ats):
    os = OutputSpec(tmp_path / "ats.xyz", tags={"test_tag": 5})
    os.write(ats)

    cs = ConfigSet(tmp_path / "ats.xyz")

    for at in cs:
        assert at.info["test_tag"] == 5
