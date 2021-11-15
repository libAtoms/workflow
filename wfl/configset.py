import glob
import os
import time
import traceback

from pathlib import Path
import tempfile

import ase.io
from ase import Atoms
from ase.io.formats import filetype as ase_filetype

try:
    from abcd import ABCD
    from abcd.database import AbstractABCD
except ModuleNotFoundError:
    ABCD = None


def _fmt(title, obj):
    # writes str or iterable and apply formatting
    if isinstance(obj, str):
        ss = obj
    else:
        ss = ", ".join(obj)
    return f'\n   {title}: {ss}'


def _parse_abcd(abcd_conn):
    if isinstance(abcd_conn, str):
        return ABCD.from_url(abcd_conn)
    elif abcd_conn is None or isinstance(abcd_conn, AbstractABCD):
        return abcd_conn
    else:
        raise RuntimeError('Got abcd_conn type {}, not None or ABCD or URL string'.format(type(abcd_conn)))


class ConfigSet_in:
    def __init__(self, abcd_conn=None,
                 file_root='', input_files=None, default_index=':',
                 input_queries=None, input_configs=None, input_configsets=None,
                 parallel_io=False, verbose=False):
        """Thin input layer for Set of atomic configurations, files or ABCD queries or list of Atoms.

        Notes
        -----
        - The input sources are mutually exclusive

        Parameters
        ----------
        abcd_conn: str / AbstractABCD, default None
            ABCD connection URL or ABCD object
        file_root: str, default ``
            path to prepend to every file name
        input_files: str / iterable(str) / 2-tuple(str) / iterable(2-tuple(str))
            Input file name globs, optionally as 2-tuples of (filename_glob, index).
            The latter only works with more than one tuple
            index here overrides separate default_index arg
        default_index: str, default `:`
            default indexing, applied to all files unless overridden by indexing specified in input_files argument
        input_queries: dict/str / iterable(dict/str)
            ABCD queries, iterated over to form list of input configs
        input_configs: ase.Atoms / iterable(Atoms) / iterable(iterable(Atoms))
            Atoms structures given directly
        input_configsets: iterable(ConfigSet_in)
            ConfigSet_in objects to be merged
        parallel_io: bool, default False
            parallel ASE atomic config input
        verbose: bool, default False
            verbose output
        """
        # is there a nicer way of testing this?
        assert sum([i is not None for i in [input_files, input_queries, input_configs, input_configsets]]) <= 1

        self.debug = False

        self.verbose = verbose
        self.parallel_io = parallel_io

        # parse ABCD URL if provided
        self.abcd = _parse_abcd(abcd_conn)

        # copy input args to self attributes
        self.file_root = file_root
        self.input_files = None
        self.input_queries = input_queries
        self.input_configs = input_configs
        self.default_index = default_index

        # properties needed elsewhere
        self.current_input_file = None

        if self.input_queries is not None:
            # fix up queries
            if self.abcd is None:
                raise RuntimeError('Got input_queries, but abcd_conn was not specified')

            if isinstance(self.input_queries, (dict, str)):
                self.input_queries = [self.input_queries]
            if self.verbose:
                print('added queries', self.input_queries)
        elif input_files is not None:
            # fix up files
            self.input_files = []
            if isinstance(input_files, str):
                # single string, one glob
                # can't accept one 2-tuple of (file, index) because there's no perfect way of distinguishing it from
                #    2-tuple of filenames (unless we want to test whether second file is in form of an index and assume
                #    it is if it matches.  The index argument can still be used in this case.
                input_files = [input_files]

            for glob_index in input_files:
                if isinstance(glob_index, str):
                    glob_index = (glob_index, self.default_index)
                try:
                    if len(glob_index) != 2:
                        raise RuntimeError(
                            'Got input_files item \'{}\' len = {} != 2'.format(glob_index, len(glob_index)))
                    if self.verbose:
                        print('adding files in ', glob_index)
                    self.input_files.extend(
                        [(f, glob_index[1]) for f in sorted(glob.glob(os.path.join(file_root, glob_index[0])))])
                except Exception as exc:
                    raise RuntimeError(('Got input_files \'{}\' with some contents type '
                                        'not str or 2-tuple of str').format(input_files)) from exc
            if len(self.input_files) == 0:
                raise RuntimeError('input glob(s) \'{}\' did not match any files'.format(input_files))
        elif self.input_configs is not None:
            # fix up configs
            if isinstance(self.input_configs, Atoms):
                self.input_configs = [[self.input_configs]]
            else:
                try:
                    if isinstance(next(iter(self.input_configs)), Atoms):
                        # iterable returning Atoms becomes a single group
                        self.input_configs = [self.input_configs]
                except TypeError as exc:
                    raise RuntimeError('input_configs type {} not Atoms or iterable'.format(
                        self.input_configs)) from exc
            if self.verbose:
                print('added queries #', [len(ats) for ats in self.input_configs])
        elif input_configsets is not None:
            if isinstance(input_configsets, ConfigSet_in):
                input_configsets = [input_configsets]
            for configset in input_configsets:
                if configset is None:
                    # allow for list that includes None, so it's easy to handle
                    # combining things that include optional arguments which may be none
                    continue
                self.merge(configset)
        # else no inputs, consider as empty


    def get_input_type(self):
        return self.input_queries is not None, self.input_files is not None, self.input_configs is not None


    def merge(self, configset):
        assert isinstance(configset, ConfigSet_in)

        if not any(configset.get_input_type()):
            # empty
            return

        if not any(self.get_input_type()):
            self.abcd = configset.abcd
            if configset.input_files is not None:
                self.input_files = configset.input_files.copy()
            if configset.input_queries is not None:
                self.input_queries = configset.input_queries.copy()
            if configset.input_configs is not None:
                self.input_configs = configset.input_configs.copy()

        elif self.get_input_type() == configset.get_input_type():
            if self.abcd != configset.abcd:
                raise RuntimeError('mismatched ABCD connections {} {}'.format(self.abcd, configset.abcd))
            if self.input_files is not None:
                self.input_files.extend(configset.input_files)
            if self.input_queries is not None:
                self.input_queries.extend(configset.input_queries)
            if self.input_configs is not None:
                self.input_configs.extend(configset.input_configs)

        else:
            raise RuntimeError(
                'input_type {} tried to merge input_type {}'.format(self.get_input_type(), configset.get_input_type))


    # iterator over all configurations in inputs
    def __iter__(self):
        self.current_input_file = None

        if self.input_queries is not None:
            for q in self.input_queries:
                for at in self.abcd.get_atoms(q):
                    yield at
        elif self.input_files is not None:
            for fin in self.input_files:
                self.current_input_file = fin[0]
                for at in ase.io.iread(fin[0], index=fin[1], parallel=self.parallel_io):
                    yield at
            self.current_input_file = None
        elif self.input_configs is not None:
            for at_group in self.input_configs:
                for at in at_group:
                    yield at


    def group_iter(self):
        self.current_input_file = None

        if self.input_queries is not None:
            for q in self.input_queries:
                yield self.abcd.get_atoms(q)
        elif self.input_files is not None:
            for fin in self.input_files:
                self.current_input_file = fin[0]
                yield ase.io.read(fin[0], index=fin[1])
            self.current_input_file = None
        elif self.input_configs is not None:
            for at_group in self.input_configs:
                yield at_group


    def get_input_files(self):
        return [fin[0] for fin in self.input_files]


    def get_current_input_file(self):
        return self.current_input_file


    def in_memory(self):
        """Create a ConfigSet_in containing the same configs, but in memory

        Returns
        -------
        ci: ConfigSet_in
        """

        if self.input_configs is not None:
            return ConfigSet_in(input_configs=self.input_configs)
        else:
            return ConfigSet_in(input_configs=list(self))


    def is_one_file(self):
        """Test if self is only one file with trivial index

        Returns
        -------
        filename: str filename, or False
        """

        if self.input_files is not None and len(self.input_files) == 1 and self.input_files[0][1] == ':':
            return self.input_files[0][0]
        else:
            return False


    def to_scratch_file(self, filename):
        """Create a ConfigSet_in containing the same configs, but in one temporary file

        Parameters
        ----------
        filename: str or Path
            filename used as template for mkstemp


        Returns
        -------
        filename: str created filename
        """

        filename = Path(filename)
        filename = tempfile.mkstemp(prefix=filename.stem+'.', suffix=filename.suffix, dir=filename.parent)[1]
        ase.io.write(filename, list(self))

        return filename


    def __str__(self):
        s = 'ConfigSet_in:'
        if self.input_configs is not None and len(self.input_configs) > 0:
            s += _fmt("input configs #", [str(len(at_group)) for at_group in self.input_configs])
        if self.input_files is not None and len(self.input_files) > 0:
            s += _fmt("input files", [f"{fn} @ {idx}" for fn, idx in self.input_files])
        if self.input_queries is not None and len(self.input_queries) > 0:
            s += _fmt("input queries", [str(q) for q in self.input_queries])

        if self.abcd is not None:
            s += _fmt('ABCD connection', str(self.abcd))
        return s


class ConfigSet_out:
    def __init__(self, abcd_conn=None, set_tags=None,
                 file_root='', output_files=None, output_abcd=False,
                 force=None, all_or_none=True, parallel_io=False, verbose=False):
        """Thin output layer for configurations into files, ABCD, or atomic configs

        Notes
        -----
            output_files and output_abcd may not both be set (but both many be unset to save to list(atoms))

        Parameters
        ----------
        abcd_conn: str / AbstractABCD, default None
            ABCD connection URL or ABCD object
        set_tags: dict
            dict of tags and value to set in every config
        file_root: str, default ``
            path to prepend to every file name
        output_files: str / dict / list(len=1) / tuple(len=1), default=None
            output file, or dict mapping from input to output files
        output_abcd: bool, default False
            write output to ABCD
        force: bool, default False
            write even if doing so will overwrite (file) or some config with set_tags already exist (ABCD)
        all_or_none: bool, default True
            write to temporary filename/tags and rename at the end, to ensure that files/tags only exist
            when output is complete
        parallel_io: bool, default False
            parallel ASE atomic config output
        verbose: bool, default False
            verbose output
        """

        if all_or_none:
            # make sure force is set correctly
            if force is None:
                force=True
            elif not force:
                raise RuntimeError(f'Cannot pass force={force} must evaluate to True with all_or_none=True')
        elif force is None:
            # default force to False
            force = False

        self.verbose = verbose
        self.parallel_io = parallel_io

        # properties needed elsewhere
        self.current_output_file = None
        self.tmp_output_files = None
        self.last_flush = time.time()

        # make sure output_files and output_abcd are not both set
        assert output_files is None or not output_abcd

        self.abcd = _parse_abcd(abcd_conn)

        # set set_tags from arguments
        if set_tags is not None and not isinstance(set_tags, dict):
            raise RuntimeError('Got set_tags type {} not dict'.format(type(set_tags)))
        self.set_tags = set_tags
        if self.verbose:
            print('setting tags', self.set_tags)

        # set atomic operation (sort-of) mode
        self.all_or_none = all_or_none

        self.output_abcd = output_abcd
        self.output_files = None
        self.output_files_map = None
        self.output_configs = None
        if self.output_abcd:
            if self.all_or_none:
                raise RuntimeError('all-or-none for database output not implemented')
            if self.abcd is None:
                raise RuntimeError('Got output_abcd, but abcd_conn was not specified')
            if self.verbose:
                print('writing to ABCD')
        elif output_files is not None:
            if isinstance(output_files, dict):
                self.output_files = [os.path.join(file_root, fout) for fout in output_files.values()]
                self.output_files_map = lambda fin: os.path.join(file_root, output_files.get(fin))
            else:
                if not isinstance(output_files, str):
                    try:
                        if len(output_files) == 1 and isinstance(next(iter(output_files)), str):
                            # extract single string from iterable
                            output_files = next(iter(output_files))
                        else:
                            raise ValueError(f'Got type {type(output_files)} of length {len(output_files)} other '
                                             f'than 1 or content type other than str, cannot map to it')
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f'Got output_files type {type(output_files)} other than iterable with '
                                           f'1 str item, str, or dict')
                self.output_files = [os.path.join(file_root, output_files)]
                self.output_files_map = lambda fin: self.output_files[0]
            if self.verbose:
                print('writing to output_files', self.output_files)
        # else:
        # no file or ABCD output, will return a list of Atoms if requested

        # overwrite if output exists
        if not force:
            self.fail_if_output_exists()

        self.pre_write()


    def is_done(self, even_if_not_all_or_none=False):
        if not even_if_not_all_or_none and not self.all_or_none:
            # if not all_or_none, do not claim that output is done, because it may be incomplete
            return False

        if self.output_abcd:
            # if any configs have requested tags, it must be done
            return self.abcd.count(self.set_tags) > 0
        if self.output_files is not None:
            # if all files exist, it must be done, since all_or_none must be set
            return all([os.path.exists(fout) for fout in self.output_files])

        # fall through to here only if output is a list of configs, never already available
        return False


    def fail_if_output_exists(self):
        # check for non-unique tags for abcd output and existing files for file output
        if self.output_abcd:
            if self.abcd.count(self.set_tags) > 0:
                raise RuntimeError('Got non-unique set_tags, pass force to override')
        if self.output_files is not None:
            for fout in self.output_files:
                if os.path.exists(fout):
                    raise RuntimeError(('Will write to file \'{}\' that exists, '
                                        'pass force to override').format(fout))


    def pre_write(self):
        self.output_configs = None
        self.current_output_file = None
        self.tmp_output_files = None

        if self.output_files is not None:
            self.tmp_output_files = []
        elif not self.output_abcd:
            # output to config list
            self.output_configs = []


    # iteratively write to place corresponding to config input iterator last returned
    def write(self, ats, from_input_file=None, flush_interval=10):
        # promote to iterable(Atoms)
        if isinstance(ats, Atoms):
            ats = [ats]

        # set tags
        if self.set_tags is not None:
            for at in ats:
                at.info.update(self.set_tags)

        if self.output_files is None and not self.output_abcd:
            # save for later return as list(Atoms) or list(list(Atoms))
            self.output_configs.append(ats)
            if self.verbose:
                print('write wrote {} configs to self.output_configs as group'.format(len(ats)))
        elif self.output_abcd:
            # save to database
            self.abcd.push(ats)
            if self.verbose:
                print('write wrote {} to {}'.format(len(ats), self.abcd))
        else:
            # write to files
            try:
                real_output_filename = self.output_files_map(from_input_file)
            except Exception as e:
                traceback.print_exc(e)
                raise RuntimeError('Failed to get output filename from map \'{}\' and input file \'{}\''.format(
                    self.output_files_map, from_input_file))

            if self.all_or_none:
                t = list(os.path.split(real_output_filename))
                # should we add something more unique, e.g. uuid4?
                t[-1] = 'tmp.' + t[-1]
                use_output_filename = os.path.join(*t)
            else:
                use_output_filename = real_output_filename

            # this assumes each output file is hit only once
            # should we think about how to deal with appending?
            if self.current_output_file is None or self.current_output_file.name != use_output_filename:
                if self.current_output_file:
                    self.current_output_file.close()
                self.current_output_file = open(use_output_filename, 'w')

                if self.all_or_none:
                    self.tmp_output_files.append((real_output_filename, use_output_filename))

            ase.io.write(self.current_output_file, ats, format=ase_filetype(self.current_output_file.name, read=False),
                         parallel=self.parallel_io)

            # flush if enough time has passed
            cur_time = time.time()
            if flush_interval >= 0 and cur_time >= self.last_flush + flush_interval:
                self.current_output_file.flush()
                self.last_flush = cur_time

            if self.verbose:
                print('write wrote {} to {}'.format(len(ats), self.current_output_file.name))


    def end_write(self):
        # end writing
        try:
            self.current_output_file.close()
        except AttributeError:
            pass

        if self.output_configs is not None or self.output_abcd:
            # configs are in array, nothing to do
            return

        # rename files if needed for all-or-none
        if self.tmp_output_files:
            if self.verbose:
                print('renaming', self.tmp_output_files)
            for real_name, tmp_name in self.tmp_output_files:
                os.rename(tmp_name, real_name)
            self.tmp_output_files = None


    # get list(Atoms) or ConfigSet reference to output
    def to_ConfigSet_in(self):
        if self.output_configs is not None and len(self.output_configs) > 0:
            return ConfigSet_in(input_configs=self.output_configs)
        elif self.output_abcd:
            if self.set_tags is None:
                # no way to find written configs without (unique) tags
                return None
            else:
                return ConfigSet_in(input_queries=self.set_tags, abcd_conn=self.abcd)
        else:
            return ConfigSet_in(input_files=self.output_files)


    def __str__(self):
        s = 'ConfigSet_out:'
        if self.set_tags is not None:
            s += _fmt('output tags', str(self.set_tags))

        if self.output_abcd:
            s += _fmt('output to ABCD', "")
        elif self.output_files is not None:
            s += _fmt('output files: ', self.output_files)

        if self.abcd is not None:
            s += _fmt('ABCD connection', str(self.abcd))
        return s
