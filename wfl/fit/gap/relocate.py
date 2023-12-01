import sys
import warnings
import shutil

from pathlib import Path
from xml.etree import cElementTree


def gap_relocate(old_file, new_file, extra_filename_glob=None, delete_old=False):
    """relocate a GAP from one file to another, changing tags and attributes as needed to
       be consistent with new name, and renaming additional files (referred to in the xml
       or just with same leading part of name

    Parameters
    ----------
    old_file: str
        original GAP xml filename
    new_file: str
        new GAP xml filename
    extra_filename_glob: str, default '*'
        glob to add to end of old_file to generate list of files that need to be renamed
    delete_old: bool, default False
        delete the old_file and all the other related files
    """

    old_file = Path(old_file)
    new_file = Path(new_file)

    if old_file == new_file:
        # nothing to do
        return

    assert old_file.exists()
    assert new_file.parent.exists()

    # need to change name as well
    old_name = old_file.name.replace('.xml', '')
    new_name = new_file.name.replace('.xml', '')

    et = cElementTree.parse(old_file)
    root = et.getroot()
    if root.tag == old_name:
        root.tag = new_name

    filenames = {}
    for child in root:
        if child.tag == 'Potential':
            if child.get('label') == old_name:
                child.set('label', new_name)
        filenames.update(_filename_rename(child, old_name, new_name))

    cleanup_files = []

    for old_xml_ref_file, new_xml_ref_file in filenames.items():
        cleanup_files.append(Path(old_file).parent / old_xml_ref_file)
        sys.stderr.write(f'copying {cleanup_files[-1]} {Path(new_file).parent / new_xml_ref_file}\n')
        shutil.copyfile(cleanup_files[-1], Path(new_file).parent / new_xml_ref_file)

    if extra_filename_glob is not None:
        for extra_file in old_file.parent.glob(old_file.name + extra_filename_glob):
            if extra_file.name != old_file.name and extra_file.name not in filenames:
                cleanup_files.append(Path(old_file).parent / extra_file.name)
                sys.stderr.write(f'copying extra {cleanup_files[-1]} '
                                 f'{Path(new_file).parent / extra_file.name.replace(old_file.name, new_file.name, 1)}\n')
                shutil.copyfile(cleanup_files[-1], Path(new_file).parent / extra_file.name.replace(old_file.name, new_file.name, 1))

    et.write(new_file)
    cleanup_files.append(old_file)

    if delete_old:
        for cleanup_file in cleanup_files:
            sys.stderr.write(f'Deleting {cleanup_file}\n')
            cleanup_file.unlink()


def _filename_rename(root, old_name, new_name):
    filenames = {}
    for k in root.attrib:
        v = root.attrib[k]
        if 'filename' in k:
            if not v.startswith(old_name):
                warnings.warn(f'Got apparent filename "{k}: {v}", but filename does not start with old GAP name "{old_name}"')
            else:
                v_new = v.replace(old_name, new_name, 1)
                assert v not in filenames
                filenames[v] = v_new
                root.attrib[k] = v_new

    for child in root:
        filenames.update(_filename_rename(child, old_name, new_name))

    return filenames


if __name__ == '__main__':
    assert len(sys.argv) == 3 or len(sys.argv) == 4
    gap_relocate(*sys.argv[1:])
