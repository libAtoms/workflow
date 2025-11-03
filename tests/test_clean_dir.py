from wfl.calculators.utils import clean_rundir

# def clean_rundir(rundir, keep_files, default_keep_files, calculation_succeeded):

all_files = ["a", "aa", "b", "c", "d"]
default_keep_files = ["a*", "b"]
actual_default_keep_files = ["a", "aa", "b"]

def create_files(dir):
    for filename in all_files:
        with open(dir / filename, "w") as fout:
            fout.write("content\n")

def check_dir(dir, files):
    if files is None:
        # even path doesn't exist
        assert not dir.is_dir()
        return

    files = set(files)

    # all expected files are present
    for file in files:
        assert (dir / file).is_file()
    # all present files are expected
    for file in dir.iterdir():
        assert file.name in files

def test_clean_rundir(tmp_path):
    # keep True
    # keep all files regardless of success
    for succ, files in [(True, all_files), (False, all_files)]:
        p = tmp_path / f"True_{succ}"
        p.mkdir()
        create_files(p)
        clean_rundir(p, True, default_keep_files, calculation_succeeded=succ)
        check_dir(p, files)

    # keep False
    # succeeded means keep nothing, failed means keep default
    for succ, files in [(True, None), (False, actual_default_keep_files)]:
        p = tmp_path / f"False_{succ}"
        p.mkdir()
        create_files(p)
        clean_rundir(p, False, default_keep_files, calculation_succeeded=succ)
        check_dir(p, files)

    # keep subset of default
    # succeeded means keep subset, failed means keep default
    for succ, files in [(True, ["a"]), (False, actual_default_keep_files)]:
        p = tmp_path / f"a_{succ}"
        p.mkdir()
        create_files(p)
        clean_rundir(p, ["a"], default_keep_files, calculation_succeeded=succ)
        check_dir(p, files)

    # keep different set from default
    # succeeded means keep set, failed means keep union of default and set
    for succ, files in [(True, ["a", "c"]), (False, actual_default_keep_files + ["a", "c"])]:
        p = tmp_path / f"ac_{succ}"
        p.mkdir()
        create_files(p)
        clean_rundir(p, ["a", "c"], default_keep_files, calculation_succeeded=succ)
        check_dir(p, files)
