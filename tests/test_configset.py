import pytest

from wfl.configset import ConfigSet
from ase.atoms import Atoms
import ase.io

@pytest.fixture
def ats():
    return [Atoms(numbers=[Z]) for Z in range(1, 1+10)]

def print_tree(items, prefix=""):
    if isinstance(items, Atoms):
        print(f"{prefix}Atoms", items.numbers, items.info.get("_ConfigSet_loc"))
    else:
        for item in items:
            if isinstance(item, Atoms):
                print(f"{prefix}Atoms", item.numbers, item.info.get("_ConfigSet_loc"))
            elif isinstance(item, ConfigSet):
                print(f"{prefix}subtree ConfigSet")
                print_tree(item.groups(), prefix + "    ")
            else:
                print(f"{prefix}subtree other")
                print_tree(item, prefix + "    ")

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

def test_single_Atoms(ats):
    print("CHECK single Atoms")
    locs = [" / 0"]
    cs = ConfigSet(ats[0])
    check_ConfigSet(cs, locs, gather_numbers([ats[0]]))

def test_list_Atoms(ats):
    print("CHECK flat list(Atoms)")
    cs = ConfigSet(ats)
    locs = [f" / {i}" for i in range(len(ats))]
    check_ConfigSet(cs, locs, gather_numbers(ats))

def test_list_list_Atoms(ats):
    print("CHECK list(list(Atoms))")
    locs = []
    for i0 in range(2):
        for i1 in range(3):
            locs.append(f" / {i0} / {i1}")
    ats_i = [ats[0:3], ats[3:6]]
    cs = ConfigSet(ats_i)
    check_ConfigSet(cs, locs, gather_numbers(ats_i))

def test_list_list_list_Atoms(ats):
    print("CHECK list(list(list(Atoms)))")
    locs = []
    for i0 in range(2):
        for i1 in range(2):
            for i2 in range(2):
                locs.append(f" / {i0} / {i1} / {i2}")
    ats_i =[[ats[0:2], ats[2:4]], [ats[4:6], ats[6:8]]]
    cs = ConfigSet(ats_i)
    check_ConfigSet(cs, locs, gather_numbers(ats_i))

def test_single_file_single_Atoms(tmp_path, ats):
    print("CHECK single file with single Atoms")
    ase.io.write(tmp_path / "ats.xyz", ats[0])
    cs = ConfigSet(tmp_path / "ats.xyz")
    locs = [" / 0"]
    check_ConfigSet(cs, locs, gather_numbers([ats[0]]))

def test_single_file_mult_Atoms(tmp_path, ats):
    print("CHECK single file with mult Atoms")
    ase.io.write(tmp_path / "ats.xyz", ats[0:5])
    cs = ConfigSet(tmp_path / "ats.xyz")
    locs = [f" / {i}" for i in range(5)]
    check_ConfigSet(cs, locs, gather_numbers(ats[0:5]))

def test_mult_files_mult_Atoms(tmp_path, ats):
    print("CHECK mult file with mult Atoms")
    ase.io.write(tmp_path / "ats_0.xyz", ats[0:5])
    ase.io.write(tmp_path / "ats_1.xyz", ats[5:10])
    cs = ConfigSet(["ats_0.xyz", "ats_1.xyz"], file_root=tmp_path)
    locs = [f" / {i0} / {i1}" for i0 in range(2) for i1 in range(5)]
    check_ConfigSet(cs, locs, gather_numbers([ats[0:5], ats[5:10]]))

def test_mult_files_mult_Atoms_glob_file(tmp_path, ats):
    print("CHECK mult file with mult Atoms using a glob for the filename")
    ase.io.write(tmp_path / "ats_0.xyz", ats[0:5])
    ase.io.write(tmp_path / "ats_1.xyz", ats[5:10])
    locs = [f" / {i0} / {i1}" for i0 in range(2) for i1 in range(5)]

    # file_root + glob in filename
    cs = ConfigSet("ats_*.xyz", file_root=tmp_path)
    check_ConfigSet(cs, locs, gather_numbers([ats[0:5], ats[5:10]]))

    # glob in full pathname
    cs = ConfigSet(tmp_path / "ats_*.xyz")
    check_ConfigSet(cs, locs, gather_numbers([ats[0:5], ats[5:10]]))

    # glob in absolute pathname
    cs = ConfigSet(tmp_path.absolute() / "ats_*.xyz")
    check_ConfigSet(cs, locs, gather_numbers([ats[0:5], ats[5:10]]))

def test_mult_files_mult_Atoms_glob_dir(tmp_path, ats):
    print("CHECK mult file with mult Atoms using a glob for directory that contains the files")
    (tmp_path / "dir_0").mkdir()
    (tmp_path / "dir_1").mkdir()
    ase.io.write(tmp_path / "dir_0" / "ats.xyz", ats[0:5])
    ase.io.write(tmp_path / "dir_1" / "ats.xyz", ats[5:10])
    locs = [f" / {i0} / {i1}" for i0 in range(2) for i1 in range(5)]

    # glob for dir name, but same filename
    cs = ConfigSet(tmp_path / "dir_*" / "ats.xyz")
    check_ConfigSet(cs, locs, gather_numbers([ats[0:5], ats[5:10]]))

    # workdir with glob for dir name, but same filename
    cs = ConfigSet("dir_*/ats.xyz", file_root=tmp_path)
    check_ConfigSet(cs, locs, gather_numbers([ats[0:5], ats[5:10]]))

def test_mult_files_mult_Atoms_mult_glob_dir(tmp_path, ats):
    print("CHECK mult file with mult Atoms using multiple globs glob for directory that contains the files")
    (tmp_path / "dir_0").mkdir()
    (tmp_path / "dir_1").mkdir()
    (tmp_path / "other_dir_0").mkdir()
    ase.io.write(tmp_path / "dir_0" / "ats.xyz", ats[0:3])
    ase.io.write(tmp_path / "dir_1" / "ats.xyz", ats[3:6])
    ase.io.write(tmp_path / "other_dir_0" / "ats.xyz", ats[6:10])
    locs = [f" / {i0} / {i1}" for i0, i1 in [(0, 0), (0, 1), (0,2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]]

    # glob for dir name, but same filename
    cs = ConfigSet([tmp_path / "dir_[01]" / "ats.xyz", tmp_path / "other_dir_*" / "ats.xyz"])
    check_ConfigSet(cs, locs, gather_numbers([ats[0:3], ats[3:6], ats[6:10]]))

    # workdir with glob for dir name, but same filename
    cs = ConfigSet(["dir_[0-1]/ats.xyz", "other_dir_*/ats.xyz"], file_root=tmp_path)
    check_ConfigSet(cs, locs, gather_numbers([ats[0:3], ats[3:6], ats[6:10]]))

def test_single_file_tree_Atoms(tmp_path, ats):
    for i in range(0, 3):
        ats[i].info["_ConfigSet_loc"] = f" / 0 / {i}"
    for i in range(3, 6):
        ats[i].info["_ConfigSet_loc"] = f" / 1 / {i - 3}"
    ase.io.write(tmp_path / "ats.xyz", ats[0:6])

    cs = ConfigSet("ats.xyz", file_root=tmp_path)
    locs = [f" / {i0} / {i1}" for i0 in range(2) for i1 in range(3)]
    check_ConfigSet(cs, locs, gather_numbers([ats[0:3], ats[3:6]]))

def test_mult_files_mult_trees_Atoms(tmp_path, ats):
    print("CHECK mult file with trees of Atoms")
    for i in range(0, 2):
        ats[i].info["_ConfigSet_loc"] = f" / 0 / {i}"
    for i in range(2, 4):
        ats[i].info["_ConfigSet_loc"] = f" / 1 / {i - 2}"
    ase.io.write(tmp_path / "ats_0.xyz", ats[0:4])
    for i in range(0, 2):
        ats[i+4].info["_ConfigSet_loc"] = f" / 0 / {i}"
    for i in range(2, 4):
        ats[i+4].info["_ConfigSet_loc"] = f" / 1 / {i - 2}"
    ase.io.write(tmp_path / "ats_1.xyz", ats[4:8])

    ats_i = [[ats[0:2], ats[2:4]], [ats[4:6], ats[6:8]]]

    locs = [f" / {i0} / {i1} / {i2}" for i0 in range(2) for i1 in range(2) for i2 in range(2)]

    cs = ConfigSet([tmp_path / "ats_0.xyz", tmp_path / "ats_1.xyz"])
    check_ConfigSet(cs, locs, gather_numbers(ats_i))

def test_out_of_order_memory(ats):
    cs = ConfigSet([ats[0:3], ats[3:6]])
    sub_cs = list(cs.groups())

    locs = [f" / {i}" for i in range(3)]
    check_ConfigSet(sub_cs[1], locs, gather_numbers(ats[3:6]))
    check_ConfigSet(sub_cs[0], locs, gather_numbers(ats[0:3]))

def test_out_of_order_one_file(tmp_path, ats):
    for at_i, at in enumerate(ats[0:6]):
        at.info["_ConfigSet_loc"] = f" / {at_i // 3} / {at_i % 3}"
    ase.io.write(tmp_path / "ats_0.extxyz", ats[0:6])
    cs = ConfigSet("ats_0.extxyz", file_root=tmp_path)
    sub_cs = list(cs.groups())

    locs = [f" / {i0}" for i0 in range(3)]
    check_ConfigSet(sub_cs[1], locs, gather_numbers(ats[3:6]))
    check_ConfigSet(sub_cs[0], locs, gather_numbers(ats[0:3]))

def test_out_of_order_mult_files(tmp_path, ats):
    ase.io.write(tmp_path / "ats_0.extxyz", ats[0:3])
    ase.io.write(tmp_path / "ats_1.extxyz", ats[3:6])
    cs = ConfigSet(["ats_0.extxyz", "ats_1.extxyz"], file_root=tmp_path)
    sub_cs = list(cs.groups())

    locs = [f" / {i0}" for i0 in range(3)]
    check_ConfigSet(sub_cs[1], locs, gather_numbers(ats[3:6]))
    check_ConfigSet(sub_cs[0], locs, gather_numbers(ats[0:3]))

def test_from_ConfigSet(tmp_path, ats):
    ats_i = [ats[0:3], ats[3:6]]
    ase.io.write(tmp_path / "ats_0.extxyz", ats[0:3])
    ase.io.write(tmp_path / "ats_1.extxyz", ats[3:6])
    locs = [f" / {i0} / {i1}" for i0 in range(2) for i1 in range(3)]

    # mult files
    cs_files = ConfigSet(["ats_0.extxyz", "ats_1.extxyz"], file_root=tmp_path)
    check_ConfigSet(cs_files, locs, gather_numbers(ats_i))
    check_ConfigSet(ConfigSet(cs_files), locs, gather_numbers(ats_i))

    # in memory
    cs_mem = ConfigSet(ats_i)
    check_ConfigSet(cs_mem, locs, gather_numbers(ats_i))
    check_ConfigSet(ConfigSet(cs_mem), locs, gather_numbers(ats_i))

    # one file
    cs_file = ConfigSet("ats_0.extxyz", file_root=tmp_path)
    locs = [f" / {i0}" for i0 in range(3)]
    check_ConfigSet(cs_file, locs, gather_numbers(ats[0:3]))
    check_ConfigSet(ConfigSet(cs_file), locs, gather_numbers(ats[0:3]))

def test_from_mult_ConfigSets(tmp_path, ats):
    ats_i_0 = ats[0:3]
    ats_i_1 = ats[3:6]
    cs_0 = ConfigSet(ats_i_0)
    cs_1 = ConfigSet(ats_i_1)
    cs_2 = ConfigSet([])

    # in memory, one group per source ConfigSet
    locs = [f" / {i0} / {i1}" for i0 in range(2) for i1 in range(3)]
    check_ConfigSet(ConfigSet([cs_0, cs_1, cs_2]), locs, gather_numbers([ats_i_0, ats_i_1]))

    # in files, flatten to one group per file
    ats_i_2 = ats[6:9]
    ase.io.write(tmp_path / "ats_0.extxyz", ats_i_0)
    ase.io.write(tmp_path / "ats_1.extxyz", ats_i_1)
    ase.io.write(tmp_path / "ats_2.extxyz", ats_i_2)
    cs_01 = ConfigSet(["ats_0.extxyz", "ats_1.extxyz"], file_root=tmp_path)
    cs_2 = ConfigSet("ats_2.extxyz", file_root=tmp_path)

    locs = [f" / {i0} / {i1}" for i0 in range(3) for i1 in range(3)]
    check_ConfigSet(ConfigSet([cs_01, cs_2]), locs, gather_numbers([ats_i_0, ats_i_1, ats_i_2]))

    # same check for overloaded + operator
    check_ConfigSet(cs_01 + cs_2, locs, gather_numbers([ats_i_0, ats_i_1, ats_i_2]))

def test_from_ConfigSets_mixed_0(tmp_path, ats):
    ats_i_0 = ats[0:3]
    ats_i_1 = ats[3:6]
    ase.io.write(tmp_path / "ats_1.extxyz", ats_i_1)
    cs_0 = ConfigSet(ats_i_0)
    cs_1 = ConfigSet("ats_1.extxyz", file_root=tmp_path)

    with pytest.raises(ValueError):
        cs = ConfigSet([cs_0, cs_1])

    cs_0 = ConfigSet(ats_i_0)
    cs_1 = ConfigSet("ats_1.extxyz", file_root=tmp_path)

    with pytest.raises(ValueError):
        cs = ConfigSet([cs_1, cs_0])
