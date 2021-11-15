import pytest
pytestmark = pytest.mark.remote

from wfl.expyre.resources import Resources

def test_cores(expyre_dummy_config):
    import wfl.expyre.config

    nodes = wfl.expyre.config.systems['_sys_default'].partitions

    assert Resources(max_time='1h', n=(16, 'tasks')).find_nodes(nodes) == ('node16_1,node16_2', {'nnodes': 1, 'tot_ntasks': 16, 'tot_ncores': 16,
                                                                                         'ncores_per_node': 16, 'ncores_per_task': 1, 'ntasks_per_node': 16 })
    assert Resources(max_time='1h', n=(32, 'tasks')).find_nodes(nodes) == ('node16_1,node16_2', {'nnodes': 2, 'tot_ntasks': 32, 'tot_ncores': 32,
                                                                                         'ncores_per_node': 16, 'ncores_per_task' : 1, 'ntasks_per_node': 16 })
    assert Resources(max_time='1h', n=(36, 'tasks')).find_nodes(nodes) == ('node36', {'nnodes': 1, 'tot_ntasks': 36, 'tot_ncores': 36,
                                                                                         'ncores_per_node': 36, 'ncores_per_task' : 1, 'ntasks_per_node': 36 })
    assert Resources(max_time='1h', n=(1, 'nodes')).find_nodes(nodes) == ('node16_1,node16_2', {'nnodes': 1, 'tot_ntasks': 16, 'tot_ncores': 16,
                                                                                          'ncores_per_node': 16, 'ncores_per_task' : 1, 'ntasks_per_node': 16 })
    assert Resources(max_time='1h', n=(1, 'nodes'), partitions='node_bigmem').find_nodes(nodes) == ('node_bigmem', {'nnodes': 1, 'tot_ntasks': 56, 'tot_ncores': 56,
                                                                                                                    'ncores_per_node': 56, 'ncores_per_task' : 1, 'ntasks_per_node': 56 })
    assert Resources(max_time='1h', n=(1, 'nodes'), partitions='.*bigmem').find_nodes(nodes) == ('node_bigmem', {'nnodes': 1, 'tot_ntasks': 56, 'tot_ncores': 56,
                                                                                                                 'ncores_per_node': 56, 'ncores_per_task' : 1, 'ntasks_per_node': 56 })

    assert Resources(max_time='1h', n=(32, 'tasks'), ncores_per_task=4).find_nodes(nodes) == ('node16_1,node16_2', {'nnodes': 8, 'tot_ntasks': 32, 'tot_ncores': 128,
                                                                                         'ncores_per_node': 16, 'ncores_per_task': 4, 'ntasks_per_node' : 4 })

    try:
        r = Resources(max_time='1h', n=(17, 'tasks')).find_nodes(nodes)
    except RuntimeError:
        pass

    assert Resources(max_time='1h', n=(17, 'tasks')).find_nodes(nodes, exact_fit=False) == ('node16_1,node16_2', {'nnodes': 2, 'tot_ntasks': 17, 'tot_ncores': 32,
                                                                                                            'ncores_per_node': 16, 'ncores_per_task': 1, 'ntasks_per_node': None })
    assert Resources(max_time='1h', n=(36, 'tasks')).find_nodes(nodes, exact_fit=False) == ('node36', {'nnodes': 1, 'tot_ntasks': 36, 'tot_ncores': 36,
                                                                                                    'ncores_per_node': 36, 'ncores_per_task': 1, 'ntasks_per_node': None })
    assert Resources(max_time='1h', n=(71, 'tasks')).find_nodes(nodes, exact_fit=False) == ('node36', {'nnodes': 2, 'tot_ntasks': 71, 'tot_ncores': 72,
                                                                                                    'ncores_per_node': 36, 'ncores_per_task': 1, 'ntasks_per_node': None })


def test_mem(expyre_dummy_config):
    import wfl.expyre.config

    nodes = wfl.expyre.config.systems['_sys_default'].partitions

    assert Resources(max_time='1h', max_mem_per_task='1tb', n=(1, 'nodes'), ncores_per_task=0).find_nodes(nodes) == ('node_bigmem', {'nnodes': 1, 'tot_ntasks': 1, 'tot_ncores': 56,
                                                                                                                    'ncores_per_node': 56, 'ncores_per_task': 56, 'ntasks_per_node': 1 })


def test_time(expyre_dummy_config):
    import wfl.expyre.config
    nodes = wfl.expyre.config.systems['_sys_timelimited'].partitions

    assert Resources(max_time='30m', n=(1, 'nodes')).find_nodes(nodes) == ('debug', {'nnodes': 1, 'tot_ntasks': 40, 'tot_ncores': 40,
                                                                               'ncores_per_node': 40, 'ncores_per_task': 1, 'ntasks_per_node' : 40 })
    assert Resources(max_time='00:30', n=(1, 'nodes')).find_nodes(nodes) == ('debug', {'nnodes': 1, 'tot_ntasks': 40, 'tot_ncores': 40,
                                                                               'ncores_per_node': 40, 'ncores_per_task': 1, 'ntasks_per_node' : 40 })
    assert Resources(max_time='1h', n=(1, 'nodes')).find_nodes(nodes) == ('debug', {'nnodes': 1, 'tot_ntasks': 40, 'tot_ncores': 40,
                                                                               'ncores_per_node': 40, 'ncores_per_task': 1, 'ntasks_per_node' : 40 })
    assert Resources(max_time='1:05:00', n=(2, 'nodes')).find_nodes(nodes) == ('standard', {'nnodes': 2, 'tot_ntasks': 80, 'tot_ncores': 80,
                                                                                      'ncores_per_node': 40, 'ncores_per_task': 1, 'ntasks_per_node' : 40 })
    assert Resources(max_time='1:05:00', n=(80, 'tasks')).find_nodes(nodes) == ('standard', {'nnodes': 2, 'tot_ntasks': 80, 'tot_ncores': 80,
                                                                                           'ncores_per_node': 40, 'ncores_per_task': 1, 'ntasks_per_node' : 40 })
    assert Resources(max_time='2-1:10:05', n=(1, 'nodes')).find_nodes(nodes) == ('standard', {'nnodes': 1, 'tot_ntasks': 40, 'tot_ncores': 40,
                                                                                        'ncores_per_node': 40, 'ncores_per_task': 1, 'ntasks_per_node' : 40 })
