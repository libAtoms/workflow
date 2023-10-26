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
    num_inputs_per_queued_job: int, default -100
        num_inputs_per_python_subprocess for each job. If negative will be multiplied by iterable_autopara_wrappable
        num_inputs_per_python_subprocess
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
    ignore_failed_jobs: bool, default False
        skip failures in remote jobs
    resubmit_killed_jobs: bool, default False
        resubmit jobs that were killed without an exit status (out of walltime or crashed),
        hoping that other parameters such as walltime or memory have been changed to make run complete this time
    hash_ignore: list(str), default []
        list of arguments to ignore when doing hash of remote function arguments to determine if it's already been done
    """
    def __init__(self, sys_name, job_name, resources, num_inputs_per_queued_job=-100, pre_cmds=[], post_cmds=[],
                 env_vars=[], input_files=[], output_files=[], header_extra=[],
                 exact_fit=True, partial_node=False, timeout=3600, check_interval=30,
                 ignore_failed_jobs=False, resubmit_killed_jobs=False, hash_ignore=[]):

        self.sys_name = sys_name
        self.job_name = job_name
        self.resources = copy.deepcopy(resources)
        self.num_inputs_per_queued_job = num_inputs_per_queued_job
        self.pre_cmds = pre_cmds.copy()
        self.post_cmds = post_cmds.copy()
        self.env_vars = env_vars.copy()
        self.input_files = input_files.copy()
        self.output_files = output_files.copy()
        self.header_extra = header_extra.copy()

        self.exact_fit = exact_fit
        self.partial_node = partial_node
        self.timeout = timeout
        self.check_interval = check_interval
        self.ignore_failed_jobs = ignore_failed_jobs
        self.resubmit_killed_jobs = resubmit_killed_jobs
        self.hash_ignore = hash_ignore.copy()


    def __str__(self):
        return (f'{self.sys_name} {self.job_name} {self.resources} {self.num_inputs_per_queued_job} {self.exact_fit} '
                f'{self.partial_node} {self.timeout} {self.check_interval}')
