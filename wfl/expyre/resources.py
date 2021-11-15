import re

from .units import time_to_sec, mem_to_kB


class Resources:
    """Resources required for a task, including time, memory, cores/nodes/tasks, and particular
    partitions.  Mainly consists of code that selects appropriate partition from the list
    associated with each System.
    """

    def __init__(self, max_time, n, ncores_per_task=1, max_mem_per_task=None, partitions=None):
        """Create Resources object
        Parameters
        ----------
        max_time: int, str
            max time for job in s (int) or time spec (str)
        n: (int, str)
            int number of tasks or nodes to use and str 'tasks' or 'nodes'
        ncores_per_task: int, default 1
            cores per task, 0 for all cores in node
        max_mem_per_task: int, str, default None
            max mem per task in kB (int) or memory spec (str)
        partitions: list(str), default None
            regexps for types of node that can be used
        """
        assert n[1] in ['nodes', 'tasks']

        self.max_time = time_to_sec(max_time)
        self.n = n
        self.ncores_per_task = ncores_per_task
        self.max_mem_per_task = mem_to_kB(max_mem_per_task)
        self.partitions = partitions
        if isinstance(self.partitions, str):
            self.partitions = [self.partitions]


    def find_nodes(self, partitions, exact_fit=True, partial_node=False):
        """find a node type that accomodates requested resources
        Parameters
        ----------
        partitions: dict
            properties of available partitions
        exact_fit: bool, default True
            only return nodes that exactly satisfy the number of tasks
        partial_node: bool, default False
            allow jobs that take less than an entire node, overrides exact_fit

        Returns
        -------
        partition: str
            name of partition selected
        dict: various quantities of node
            nnodes: int, total number of nodes needed
            tot_ncores: int, total number of cores needed
            ncores_per_node: int, number of cores per node for selected nodes
            tot_ntasks: int, total number of (mpi) tasks to run
            ncores_per_task: int, cores per task
            ntasks_per_node: int, tasks per node (only if exact_fit=True, otherwise None)
        """
        selected_partitions = []

        if partial_node:
            exact_fit=False

        for partition, node_spec in partitions.items():
            if self.partitions is not None and all([re.search('^'+nt_re+'$', partition) is None for nt_re in self.partitions]):
                # wrong node type
                continue

            if node_spec['max_time'] is not None and self.max_time > node_spec['max_time']:
                # too much time
                continue

            if exact_fit and self.n[1] == 'tasks' and (self.n[0] * self.ncores_per_task) % node_spec['ncores'] != 0:
                # wrong number of cores
                continue

            if self.max_mem_per_task is not None and node_spec['max_mem'] is not None:
                nnodes, tot_ntasks, _ = self._get_nnodes_ntasks_ncores_per_task(node_spec)
                max_mem_per_node = self.max_mem_per_task * tot_ntasks / nnodes
                if max_mem_per_node > node_spec['max_mem']:
                    # too much memory
                    continue

            selected_partitions.append(partition)

        if len(selected_partitions) == 0:
            raise RuntimeError(f'Failed to find acceptable node type '
                               f'for {self} with exact_fit={exact_fit}')

        if len(selected_partitions) > 1:
            wasted_cores = []
            for nt in selected_partitions:
                node_spec = partitions[nt]
                _, tot_ntasks, ncores_per_task = self._get_nnodes_ntasks_ncores_per_task(node_spec)
                wasted_cores.append((node_spec['ncores'] - (tot_ntasks * ncores_per_task) % node_spec['ncores']) % node_spec['ncores'])

            try:
                # look for first one that matches exactly
                partition_i = wasted_cores.index(0)
            except ValueError:
                # pick best among remaining
                max_extra = min(wasted_cores)
                partition_i = wasted_cores.index(max_extra)
            selected_partitions = [selected_partitions[partition_i]]

        partition = selected_partitions[0]

        nnodes, tot_ntasks, ncores_per_task = self._get_nnodes_ntasks_ncores_per_task(partitions[partition])

        if not exact_fit:
            ntasks_per_node = None
        else:
            ntasks_per_node = int(tot_ntasks // nnodes)

        # by default use entire nodes
        ncores_per_node = partitions[partition]['ncores']
        tot_ncores = nnodes * ncores_per_node
        if partial_node and tot_ntasks * ncores_per_task < partitions[partition]['ncores']:
            # partial node
            tot_ncores = tot_ntasks * ncores_per_task
            ncores_per_node = tot_ncores

        return partition, {'nnodes': nnodes, 'tot_ncores': tot_ncores,
                           'ncores_per_node': ncores_per_node,
                           'tot_ntasks': tot_ntasks, 'ncores_per_task': ncores_per_task, 'ntasks_per_node': ntasks_per_node}


    def _get_nnodes_ntasks_ncores_per_task(self, node_spec):
        if self.ncores_per_task == 0:
            # one task per node
            nnodes = self.n[0]
            tot_ntasks = nnodes
            ncores_per_task = node_spec['ncores']

            return nnodes, tot_ntasks, ncores_per_task

        if self.n[1] == 'nodes':
            # fill up requested # of nodes
            nnodes = self.n[0]
            tot_ntasks = int(nnodes * node_spec['ncores'] // self.ncores_per_task)
        elif self.n[1] == 'tasks':
            # determine how many nodes are necessary
            tot_ntasks = self.n[0]
            nnodes = int(tot_ntasks * self.ncores_per_task // node_spec['ncores'])
            if nnodes * node_spec['ncores'] < tot_ntasks * self.ncores_per_task:
                nnodes += 1
        else:
            raise ValueError(f'number of unknown quantity {self.n[1]}, not "nodes" or "tasks"')

        return nnodes, tot_ntasks, self.ncores_per_task


    def __repr__(self):
        return (f'time={self.max_time} n={self.n} ncores_per_task={self.ncores_per_task} '
                f'mem={self.max_mem_per_task} partitions={self.partitions}')
