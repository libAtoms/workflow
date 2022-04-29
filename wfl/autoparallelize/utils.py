import itertools
import copy


class RemoteInfo:
    """Create a RemoteInfo object

    Parameters
    ----------
    sys_name: str
        name of system to run on
    job_name: str
        name for job (unique within this project)
    resources: dict or Resources
        expyre.resources.Resources or kwargs for its constructor
    job_chunksize: int, default -100
        chunksize for each job. If negative will be multiplied by iterable_autopara_wrappable chunksize
    pre_cmds: list(str)
        commands to run before starting job
    post_cmds: list(str)
        commands to run after finishing job
    env_vars: list(str)
        environment variables to set before starting job
    input_files: list(str)
        input_files to stage in starting job
    output_files: list(str)
        output_files to stage out when job is done
    header_extra: list(str), optional
        extra lines to add to queuing system header
    exact_fit: bool, default True
        require exact fit to node size
    partial_node: bool, default True
        allow jobs that take less than a whole node, overrides exact_fit
    timeout: int
        time to wait in get_results before giving up
    check_interval: int
        check_interval arg to pass to get_results
    skip_failures: bool, default False
        skip failures in remote jobs
    """
    def __init__(self, sys_name, job_name, resources, job_chunksize=-100, pre_cmds=[], post_cmds=[],
                 env_vars=[], input_files=[], output_files=[], header_extra=[],
                 exact_fit=True, partial_node=False, timeout=3600, check_interval=30,
                 skip_failures=False):

        self.sys_name = sys_name
        self.job_name = job_name
        self.resources = copy.deepcopy(resources)
        self.job_chunksize = job_chunksize
        self.pre_cmds = pre_cmds.copy()
        self.post_cmds = post_cmds.copy()
        self.env_vars = ["WFL_AUTOPARA_NPOOL=${EXPYRE_NTASKS_PER_NODE}",
                         "OMP_NUM_THREADS=${EXPYRE_NCORES_PER_TASK}"] + env_vars
        self.input_files = input_files.copy()
        self.output_files = output_files.copy()
        self.header_extra = header_extra.copy()

        self.exact_fit = exact_fit
        self.partial_node = partial_node
        self.timeout = timeout
        self.check_interval = check_interval
        self.skip_failures = skip_failures


    def __str__(self):
        return f'{self.sys_name} {self.job_name} {self.resources} {self.job_chunksize} {self.exact_fit} {self.partial_node} {self.timeout} {self.check_interval}'


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
