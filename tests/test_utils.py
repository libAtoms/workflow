from wfl.utils import misc
from wfl.utils.replace_eval_in_strs import replace_eval_in_strs


def test_chunks():
    a = range(20)

    for chunk in misc.chunks(a, 5):
        assert len(chunk) == 5


def test_replace_eval_in_strs():
    v0 = 2
    v1 = 1.03
    d = { 'a': 5, 'b': '_EVAL_no_eval' , 'c': [ '_EVAL_ {v0}*2', '_EVAL_ {v1}*3' ] }
    dref = { 'a': 5, 'b': '_EVAL_no_eval' , 'c': [ 4, 3.1 ] }

    drepl = replace_eval_in_strs(d, {'v0': v0, 'v1': v1}, n_float_sig_figs=2)

    assert drepl == dref
