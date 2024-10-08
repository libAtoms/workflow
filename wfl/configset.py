import sys
import re

import glob
from pathlib import Path


from ase.atoms import Atoms
import ase.io
import ase.io.formats

class ConfigSet:
    """Abstraction layer for storing and looping through collections of
    atomic configurations, either in memory or in files. Ignores existing
    Atom.info["_ConfigSet_loc"] fields in Atoms objects that are passed in

    If created from a list of Atoms, potentially nested, the stored nested structure
    (which is returned by the iterators) is that of the nested lists. If created from one
    file (str or Path), the top level of nesting returned by the iterators corresponds
    to individual Atoms or sub-lists within the file.  If created from a list of files
    (even of length 1), the top level of nesting returned by the iterators corresponds
    to each file.

    Iterating over the ConfigSet returns a flattened list of configurations in the object,
    with information about the nested structure (necessary for reproducing it in the
    output, see OutputSpec.store() below) available from the ConfigSet's cur_loc
    property or in the returned returned objects in Atoms.info["_ConfigSet_loc"].

    Parameters
    ----------

    items: Atoms / list(Atoms) / list(list...(Atoms)) / str / Path / list(str) / list(Path)
        configurations to store, or list of file name globs for the configurations.

    file_root: str / Path. default None
        path component to prepend to all file names
    read_kwargs: dict, default {"index": ":", "parallel": False}
        optional kwargs passed to ase.io.iread function, overriding these default values
    """

    _loc_sep = " / "

    def __init__(self, items, *, file_root=None, read_kwargs={}, _open_reader=None, _cur_at=None, _file_loc=None, _enclosing_loc=""):
        # internal use arguments:
        #
        # _open_reader: already open file reader, used to speed up reading of specific nested structure in files
        # _cur_at: current atoms object, used to speed up reading of specific nested structure in files
        # _file_loc: ConfigSet_loc used to restrict reading in an open file to a particular nesting level
        # _enclosing_loc: ConfigSet_loc that contains this ConfigSet, used to preserve nesting structure when
        #    ConfigSet.groups() returns a ConfigSet corresponding to a particular nesting level (used by
        #    wfl.autopara.utils.items_inputs_generator)

        # deal with private arguments
        if sum([_open_reader is None, _cur_at is None, _file_loc is None]) not in (0, 3):
            raise ValueError(f"Need either both or neither of _open_reader {_open_reader} _file_loc {_file_loc}")
        self._open_reader = _open_reader
        self._cur_at = _cur_at if _cur_at is not None else [None]
        self._file_loc = _file_loc if _file_loc is not None else ""
        self._enclosing_loc = _enclosing_loc
        file_root = Path(file_root) if file_root is not None else Path("")

        self.read_kwargs = {"index": ":", "parallel": False}
        self.read_kwargs.update(read_kwargs)

        # self.items can be
        #   Path or list(Path) or list(...(list(Atoms)))
        if items is None or (isinstance(items, (list, tuple)) and len(items) == 0):
            self.items = None
        elif isinstance(items, OutputSpec):
            # previously supported, so give explicit error message
            raise RuntimeError("ConfigSet(OutputSpec) not supported - use OutputSpec.to_ConfigSet()")
        elif isinstance(items, ConfigSet):
            if items._file_loc != "":
                raise ValueError("ConfigSet from ConfigSet cannot have _file_loc set")
            if isinstance(items.items, list):
                # NOTE: should any other possible content types be copied?
                self.items = items.items.copy()
            else:
                self.items = items.items
        elif isinstance(items, Atoms):
            self.items = [items]
        elif isinstance(items, (str, Path)):
            if file_root != Path("") and Path(items).is_absolute():
                raise ValueError(f"Got file_root but file {items} is an absolute path")
            # single item, could be a simple filename or a glob. Former needs to be stored as
            # Path, latter as list(Path)
            items_expanded = [Path(f) for f in sorted(glob.glob(str(file_root / items), recursive=True))]
            if len(items_expanded) == 1 and file_root / items == items_expanded[0]:
                self.items = file_root / items
            else:
                self.items = items_expanded
        elif isinstance(items[0], (str, Path)):
            self.items = []
            for file_path in items:
                assert isinstance(file_path, (str, Path))
                if file_root != Path("") and Path(file_path).is_absolute():
                    raise ValueError(f"Got file_root but file {file_path} is an absolute path")
                self.items.extend([Path(f) for f in sorted(glob.glob(str(file_root / file_path), recursive=True))])
        elif isinstance(items[0], ConfigSet):
            self.items = []
            for item in items:
                assert isinstance(item, ConfigSet)
                if item._file_loc != "":
                    raise ValueError("ConfigSet from ConfigSet cannot have _file_loc set")
                if item.items is None:
                    # empty ConfigSet, skip
                    continue
                elif isinstance(item.items, (str, Path)) or isinstance(item.items[0], (str, Path)):
                    # item contains Path(s)
                    if len(self.items) > 0 and not isinstance(self.items[-1], Path):
                        raise ValueError("Got ConfigSet containing Path after one that does not")

                    if isinstance(item.items, (str, Path)):
                        self.items.append(item.items)
                    else:
                        self.items.extend(item.items)
                else:
                    # item contains list(s)
                    if len(self.items) > 0 and isinstance(self.items[0], Path):
                        raise ValueError("Got ConfigSet containing Atoms after one that contains Path")
                    self.items.append(item.items.copy())
        else:
            # WARNING: expecting list(list(... (Atoms))), and all lists same depth,
            # but no error checking here
            self.items = items

        self._cur_loc = None


    @property
    def cur_loc(self):
        """When looping over ConfigSet, which returns a flattened list of configurations,
        current location string, which can be passed to OutputSpec.store() to
        ensure that outputs retain same nesting structure as inputs. Alternative to
        Atoms.info["_ConfigSet_loc"] in the most recently-returned Atoms object"""

        return self._cur_loc


    def __str__(self):
        """Convert to string
        """
        out = "ConfigSet with"

        if isinstance(self.items, Path):
            out += f" single file {self.items} {self._file_loc}"
        else:
            if self.items is None:
                out += " no content"
            elif isinstance(self.items[0], Path):
                out += f" {len(self.items)} files"
            elif isinstance(self.items[0], Atoms):
                out += f" {len(self.items)} Atoms"
            else:
                out += f" {len(self.items)} sub-lists"

        return out


    def __iter__(self):
        if self.items is None:
            return

        # special handling for ConfigSet generated by groups() iterator, which 
        # sets self._cur_at, self._file_loc, and self._open_reader
        if self._file_loc is not None and len(self._file_loc) > 0 and self._cur_at[0] is not None:
            assert self._open_reader is not None
            while self._cur_at[0].info["_ConfigSet_loc"].startswith(self._file_loc):
                yield self._cur_at[0]
                try:
                    self._cur_at[0] = next(self._open_reader)
                except StopIteration:
                    break
            return

        if isinstance(self.items, Path):
            # yield Atoms from one file
            ## print("DEBUG __iter__ one file", self.items)
            for at_i, at in enumerate(ase.io.iread(self.items, **self.read_kwargs)):
                loc = at.info.get("_ConfigSet_loc", ConfigSet._loc_sep + str(at_i))
                if len(self._file_loc) == 0 or re.match(self._file_loc + r'\b', loc):
                    ## print("DEBUG matching loc", loc, "file_loc", self._file_loc)
                    loc = loc.replace(self._file_loc, "", 1)
                    ## print("DEBUG stripped loc", loc)
                    at.info["_ConfigSet_loc"] = loc
                    self._cur_loc = at.info.get("_ConfigSet_loc")
                    yield at
        elif isinstance(self.items[0], Path):
            # yield Atoms from each file, prepending number of file to current loc
            for file_i, filepath in enumerate(self.items):
                for at_i, at in enumerate(ase.io.iread(filepath, **self.read_kwargs)):
                    loc = ConfigSet._loc_sep + str(file_i) + at.info.get("_ConfigSet_loc", ConfigSet._loc_sep + str(at_i))
                    if len(self._file_loc) == 0 or re.match(self._file_loc + r'\b', loc):
                        loc = loc.replace(self._file_loc, "", 1)
                        at.info["_ConfigSet_loc"] = loc
                        self._cur_loc = at.info.get("_ConfigSet_loc")
                        yield at
        else:
            # list of Atoms or of lists of ... Atoms
            # yield atoms from ConfigSet._flat_iter
            for at in ConfigSet._flat_iter(self.items):
                self._cur_loc = at.info.get("_ConfigSet_loc")
                yield at

        self._cur_loc = None


    def groups(self):
        """Generator returning a sequence of Atoms, or a sequence of
        ConfigSets, one for each sub-list.  Nesting structure reflects the
        input, as described in the ConfigSet class/constructor docstring.
        If items argument to constructor was a single file, iterator
        will return individual configs, or ConfigSets for sub-lists.
        If argument was a list of files, iterator will return a sequence
        of ConfigSets, one for the content of each file.  If argument
        was an Atoms or (nested) list of Atoms, iterator will reconstruct
        top level of initial nested structure.
        """

        if self.items is None:
            return

        # Need to keep track of cur_at_i only if looping through file without Atoms.info["_ConfigSet_loc"]
        # fields, which can never happen on more deeply nested data, so there's no need to be
        # able to transfer that to nested iterators. Therefore, keep in a local variable.
        cur_at_i = 0

        def advance(at_i=None):
            self._cur_at[0] = next(self._open_reader)
            return self._cur_at[0].info.get("_ConfigSet_loc", ConfigSet._loc_sep + str(at_i) if at_i is not None else None)

        if isinstance(self.items, Path):
            ## print("DEBUG groups() for one file self._open_reader", self._open_reader, "self._cur_at", self._cur_at) ##DEBUG
            # one file, return a ConfigSet for each group, or a sequence of individual Atoms
            if self._open_reader is None or self._cur_at[0] is None:
                # initialize reading of file
                ## print("DEBUG initializing reader", self.items) ##DEBUG
                self._open_reader = ase.io.iread(self.items, **self.read_kwargs)
                self._cur_at = [None]
                ## print("DEBUG setting initial self._cur_at = [None]") ##DEBUG
                try:
                    ## print("DEBUG advancing, getting at_loc with cur_at_i", cur_at_i) ##DEBUG
                    at_loc = advance(cur_at_i)
                    ## print("DEBUG got at_loc", at_loc) ##DEBUG
                except StopIteration:
                    ## print("DEBUG got EOF, returning") ##DEBUG
                    # indicate EOF
                    self._cur_at = [None]
                    return
            else:
                # just get at_loc from current atoms object
                at_loc = self._cur_at[0].info.get("_ConfigSet_loc")

            try:
                # Skip any that don't match self._file_loc
                ## print("DEBUG first skipping non-matching, at_loc", at_loc, "self._file_loc", self._file_loc) ##DEBUG
                while not at_loc.startswith(self._file_loc):
                    at_loc = advance()
                ## print("DEBUG after first skipping non-matching, new at_loc", at_loc) ##DEBUG
            except StopIteration:
                ## print("DEBUG starting second skipping") ##DEBUG
                # Failed to find config that matches self._file_loc from current position of reader.
                # Search again from start (in case we're doing something out of order)
                self._open_reader = ase.io.iread(self.items, **self.read_kwargs)
                try:
                    at_loc = advance()
                    # Skip any that don't match self._file_loc
                    while not at_loc.startswith(self._file_loc):
                        at_loc = advance()
                except StopIteration as exc:
                    # got to EOF after restaring at beginning of file, must not have
                    # any matching configs
                    raise RuntimeError(f"No matching configs in file {self.items} for location {self._file_loc}") from exc

            ## print("DEBUG after possibly skipping, now should be at right place, at_loc", at_loc, "self._file_loc", ##DEBUG
                  ## self._file_loc, "cur_at_i", cur_at_i, "self._cur_at.numbers", self._cur_at[0].numbers) ##DEBUG
            # now self._cur_at should be first config that matches self._file_loc
            requested_depth = len(self._file_loc.split(ConfigSet._loc_sep))
            ## print("DEBUG requested_depth", requested_depth) ##DEBUG
            if len(at_loc.split(ConfigSet._loc_sep)) == requested_depth + 1:
                ## print("DEBUG in right container, starting to yield atoms") ##DEBUG
                # in right container, yield Atoms
                while at_loc.startswith(self._file_loc):
                    if "_ConfigSet_loc" in self._cur_at[0].info:
                        del self._cur_at[0].info["_ConfigSet_loc"]
                    ## print("DEBUG   in loop, yielding Atoms") ##DEBUG
                    yield self._cur_at[0]
                    cur_at_i += 1
                    try:
                        at_loc = advance(cur_at_i)
                    except StopIteration:
                        ## print("DEBUG   EOF while yielding atoms") ##DEBUG
                        # indicate EOF
                        self._cur_at[0] = None
                        return
                return
            else:
                ## print("DEBUG right place, but need to go deeper") ##DEBUG
                # at first config that matches self._file_loc, but there are deeper levels to nest into

                while at_loc.startswith(self._file_loc):
                    # get location of deeper iterator from at_loc down to one deeper than requested at this level
                    new_file_loc = ConfigSet._loc_sep.join(at_loc.split(ConfigSet._loc_sep)[0:requested_depth + 1])
                    ## print("DEBUG making and yielding ConfigSet with new _file_loc", new_file_loc) ##DEBUG
                    cs_out = ConfigSet(self.items, _open_reader=self._open_reader, _cur_at=self._cur_at, _file_loc=new_file_loc,
                                       _enclosing_loc=new_file_loc)
                    ## print("DEBUG yielding ConfigSet", cs_out, "_open_reader", cs_out._open_reader, "_cur_at", self._cur_at) ##DEBUG
                    yield cs_out
                    ## print("DEBUG after yield, got self._cur_at", self._cur_at[0].numbers if self._cur_at[0] is not None else None) ##DEBUG
                    if self._cur_at[0] is None:
                        ## print("DEBUG got EOF, returning") ##DEBUG
                        # got EOF deeper inside, exit
                        return
                    # self._cur_at could have advanced when calling function iterated over yielded
                    # iterator, so update local at_loc
                    at_loc = self._cur_at[0].info.get("_ConfigSet_loc")
                    # if the iterator that was returned wasn't immediately (or at least not completely) used, at_loc
                    # will still point to the same config, skip all the ones that should have been used
                    while at_loc.startswith(new_file_loc):
                        try:
                            at_loc = advance()
                        except StopIteration:
                            ## print("DEBUG past yielded iterator got EOF") ##DEBUG
                            self._cur_at[0] = None
                            return
                    # now we should be past configs that could have been consumed by previously yielded iterator

                    ## print("DEBUG end of loop, now at_loc is", at_loc) ##DEBUG

        else:
            # self.items is list(Atoms) or list(...list(Atoms)) or list(Path)
            for item_i, item in enumerate(self.items):
                if isinstance(item, Atoms):
                    # yield each Atoms object
                    if "_ConfigSet_loc" in item.info:
                        del item.info["_ConfigSet_loc"]
                    yield item
                else:  # item must be sublist or Path
                    # yield a ConfigSet for each file or sublist
                    yield ConfigSet(item, _enclosing_loc=self._enclosing_loc + ConfigSet._loc_sep + str(item_i))


    def one_file(self):
        """Returns info on whether ConfigSet consists of exactly one file

        Returns
        -------

        one_file_name: Path of the one file, or False otherwise
        """
        if self.items is not None:
            if isinstance(self.items, Path):
                return self.items
            elif len(self.items) == 1 and isinstance(self.items[0], Path):
                return self.items[0]
            else:
                return False
        else:
            return False


    @staticmethod
    def _flat_iter(items):
        """Generator returning a flattened list of Atoms starting from a tree of nested lists
        containing only Atoms as leaves. Stores original nested tree structure in
        Atoms.info["_ConfigSet_loc"].

        Parameters
        ----------

        items: list(...(list(Atoms)))
            list, potentially nested, of Atoms
        """
        for item_i, item in enumerate(items):
            if item is None:
                continue

            if isinstance(item, Atoms):
                # set location info and yield
                item.info["_ConfigSet_loc"] = ConfigSet._loc_sep + str(item_i)
                yield item
            else:
                # make flat iterator for contained Atoms
                sub_configs = ConfigSet._flat_iter(item)
                for at in sub_configs:
                    # concatenate current location info to that already stored and yield
                    at.info["_ConfigSet_loc"] = ConfigSet._loc_sep + str(item_i) + at.info["_ConfigSet_loc"]
                    yield at


    def __add__(self, other):
        if not isinstance(other, ConfigSet):
            raise TypeError(f"unsupported operand type(s) for +: 'ConfigSet' and '{type(other)}'")
        return ConfigSet([self, other])


class OutputSpec:
    """Abstraction for writing to a ConfigSet, preserving tree structure.

    Parameters
    ----------

    files: str / Path / iterable(str / Path), default None
        list of files to store configs in, or store in memory if None

    file_root: str / Path, default None
        root directory relative to which all files will be taken

    overwrite: str, default False
        Overwrite already existing files. Default False, but note that many functions,
        including any wrapped by autoparallelize, will actually reuse existing output if
        all of it appears to be present.

    flush: bool, default True
        flush output after every write

    write_kwargs: dict
        optional extra kwargs to ase.io.write

    tags: dict
        dict of extra Atoms.info keys to set in written configs
    """
    def __init__(self, files=None, *, file_root=None, overwrite=False, flush=True, write_kwargs={}, tags={}):
        self.files = files
        self.configs = None
        self.file_root = Path(file_root if file_root is not None else "")
        self.flush = flush
        self.overwrite = overwrite
        self.write_kwargs = write_kwargs.copy()
        self.tags = tags.copy()

        if self.files is not None:
            # store in file(s)
            if isinstance(self.files, (str, Path)):
                self.single_file = True
                self.files = [self.files]
            else:
                self.single_file = False

            absolute_files = [f for f in self.files if Path(f).is_absolute()]
            if len(absolute_files) > 0 and self.file_root != Path(""):
                raise ValueError(f"Got file_root {file_root} but files {absolute_files} are absolute paths")
            self.files = [Path(f) for f in self.files]

            # wipe tmp files
            for f in self.files:
                tmp_f = self.file_root / f.parent / ("tmp." + f.name)
                tmp_f.unlink(missing_ok=True)
        else:
            # store in memory
            self.configs = []

        self.cur_file_ind = None
        self.cur_file = None

        self.closed = False

        self.first_store_call = True
        self.cur_store_loc = None


    def _existing_output_files(self):
        return [self.file_root / f for f in self.files if (self.file_root / f).exists()]


    def write(self, configs):
        """Write a set of configurations to this OutputSpec

        Parameters
        ----------
        configs: iterable(Atoms)
            Configurations to write.  If ConfigSet, location will be saved
        """
        if not self.overwrite and self.all_written():
            sys.stderr.write('Reusing existing output instead of writing ConfigSet contents since overwrite=False and output is done\n')
            return

        for at in configs:
            self.store(at, at.info.get("_ConfigSet_loc"))
        self.close()


    def store(self, configs, input_CS_loc=""):
        """Store Atoms or iterable containing Atoms or other iterables in a form that can be
        used to create a ConfigSet.  If output ConfigSet is to have same structure as input
        ConfigSet, configs must come in the same order as the input ConfigSet's flat iterator,
        with the store loc containing the _ConfigSet_loc values returned by that iterator.

        This function is generally called by wfl's built-in autoparallelization functions.
        An example that does the same thing is shown in the following construction:

        .. code-block:: python

            cs = ConfigSet(["in0.xyz", "in1.xyz"])
            os = OutputSpec(["out0.xyz", "out1.xyz"])
            for at_in in cs:
                # define at_out, either an Atoms or a (nested) list of Atoms, based on at_in
                .
                .
                # the next line can also use at_in.info["_ConfigSet_loc"] instead of cs.get_loc()
                os.store(at_out, cs.get_loc())
            os.close()


        Parameters
        ----------

        configs: Atoms / iterable(Atoms / iterable)
            configurations to store

        input_CS_loc: str, default ""
            location in input iterable (source ConfigSet) so that output iterable has same structure as source
            ConfigSet. Available from the ConfigSet that is's being iterated over via ConfigSet.cur_loc
            or the ConfigSet iterator's returned Atoms.info["_ConfigSet_loc"]. Required when writing
            to multiple files (since top level of location indicates which file), otherwise defaults
            to appending to single file or top level list of configs.
        """
        if self.closed:
            raise ValueError("Cannot store after OutputSpec.close() has been called")

        if configs is None:
            return

        if self.files is not None:
            if self.first_store_call and not self.overwrite:
                existing_files = self._existing_output_files()
                if len(existing_files) > 0:
                    raise FileExistsError(f"OutputSpec overwrite is false but output file(s) {existing_files} already exist")

            if self.single_file:
                # write to a file, preserving all location info

                # test for inconsistent sequence of provided/absent input_CS_loc
                if input_CS_loc is None or len(input_CS_loc) == 0:
                    # nothing provided, make sure this is consistent with previous calls and if so increment loc
                    if self.first_store_call:
                        self.cur_store_loc = 0
                    else:
                        # not the first call, cur_store_loc should be value from previous call
                        if self.cur_store_loc is None:
                            raise RuntimeError("got no input_CS_loc but previous calls did provide input_CS_loc")
                        self.cur_store_loc += 1
                else:
                    # provided input_CS_loc, make sure previous calls didn't omit this info
                    if self.cur_store_loc is not None:
                        raise RuntimeError("store cannot get input_CS_loc after calls that did not provide it")

                if self.cur_store_loc is not None:
                    input_CS_loc = ConfigSet._loc_sep + str(self.cur_store_loc)

                self._open_file(0)
                self._write_to_file(configs, input_CS_loc)
                self.first_store_call = False
                return

            # write to multiple files, using top level location as index to file
            if input_CS_loc is None or len(input_CS_loc) == 0:
                raise ValueError("For OutputSpec with multiple files, store() requires input_CS_loc")
            file_ind = int(input_CS_loc.split(ConfigSet._loc_sep)[1])
            self._open_file(file_ind)
            # write to file, stripping off leading component of input_CS_loc since that corresponds to
            # which file
            sub_loc = input_CS_loc.split(ConfigSet._loc_sep)[2:]
            if len(sub_loc) > 0:
                sub_loc = [""] + sub_loc
            self._write_to_file(configs, ConfigSet._loc_sep.join(sub_loc))
        else:
            # store in self.configs
            if input_CS_loc is None or len(input_CS_loc) == 0:
                # no location, just write to top level list
                self.configs.append(configs)
                return

            # store in correct location
            input_CS_loc = input_CS_loc.split(ConfigSet._loc_sep)[1:]

            # NOTE: this will create extra empty containers if indices skip.
            # We should think about whether this is the best behavior.
            cur_container = self.configs
            for depth in range(len(input_CS_loc)):
                ind = input_CS_loc[depth]
                try:
                    ind = int(ind)
                    # append empty containers as needed
                    cur_container += [[] for _ in range(ind + 1 - len(cur_container))]
                    if depth < len(input_CS_loc) - 1:
                        cur_container = cur_container[ind]
                    else:
                        cur_container[ind] = configs
                except ValueError as exc:
                    raise RuntimeError("ConfigSet_loc indices must be integers") from exc


    def close(self):
        """Finishes OutputSpec writing, closing all open files and renaming temporaries
        """
        self.closed = True

        if self.files is not None:
            if self.cur_file is not None:
                self.cur_file.close()
            for f in self.files:
                tmp_f = self.file_root / f.parent / ("tmp." + f.name)
                if tmp_f.exists():
                    tmp_f.rename(self.file_root / f)


    def all_written(self):
        """Determine if all output has been created and writing operation is done from
        a previous run, even before any configurations have been written.  Never true
        for in-memory storage.

        Since files are initially written to under temporary names and renamed to
        their final names only after OutpuSpec.close(), this will only return true if
        close has been called.

        NOTE: This will return false if a file specified in the constructor does not exist,
        which might happen if no input_CS_loc passed to OutputSpec.store() specified
        that file.
        """
        if self.files is not None:
            return all([(self.file_root / f).exists() for f in self.files])
        else:
            return False


    def to_ConfigSet(self):
        if self.files is not None:
            if self.single_file:
                cs = ConfigSet(self.files[0], file_root=self.file_root)
            else:
                cs = ConfigSet(self.files, file_root=self.file_root)
        else:
            cs = ConfigSet(self.configs)
        return cs


    def _write_to_file(self, configs, store_loc_stem):
        """Write one or more Atoms to a file, storing their locations

        Parameters
        ----------

        configs: Atoms / iterable(Atoms / iterable(Atoms / iterable))
            configurations to write

        store_loc_stem: str
            stem part of stored location to which relative location within this iterable
            will be appended
        """
        if isinstance(configs, Atoms):
            if store_loc_stem is not None and len(store_loc_stem) > 0:
                configs.info["_ConfigSet_loc"] = store_loc_stem
            configs.info.update(self.tags)
            ase.io.write(self.cur_file, configs, **self._cur_write_kwargs)
            if self.flush:
                self.cur_file.flush()
        else:
            # iterable, possibly nested
            for item_i, item in enumerate(configs):
                # WARNING: this will fail if store_loc_stem is None.  Is this what we want?
                item_loc = store_loc_stem + ConfigSet._loc_sep + str(item_i)
                self._write_to_file(item, item_loc)


    def _open_file(self, file_ind):
        """Open a file, using a temporary name, if it's not already open

        Parameters
        ----------

        file_ind: int
            index into self.files list for file to open
        """

        if self.cur_file_ind == file_ind:
            return

        # open new file
        if self.cur_file is not None:
            self.cur_file.close()

        self.cur_file_ind = file_ind

        tmp_filename = self.file_root / self.files[self.cur_file_ind].parent / ("tmp." + self.files[self.cur_file_ind].name)

        self._cur_write_kwargs = self.write_kwargs.copy()
        if "format" not in self._cur_write_kwargs:
            self._cur_write_kwargs["format"] = ase.io.formats.filetype(tmp_filename, read=False)
        self.cur_file = open(tmp_filename, "a")
        if self.flush:
            self.cur_file.flush()
