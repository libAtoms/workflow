import os, shutil, sys

import numpy as np
import pytest

@pytest.mark.skipif(not shutil.which("gap_fit"), reason="gap_fit not in PATH")  # skips it if gap_fit not in path
def test_batch_gap_fit():
    example_dir = os.path.join(
        os.path.dirname(__file__), '../', 'examples', 'iterative_gap_fit'
    )
    sys.path.append(example_dir)
    import batch_gap_fit

    batch_gap_fit.main(max_count=1, verbose=True)

    assert os.path.exists(os.path.join(example_dir, 'GAP'))
    assert os.path.exists(os.path.join(example_dir, 'MD'))
    assert os.path.exists(os.path.join(example_dir, 'errors.json'))
    assert os.path.exists(os.path.join(example_dir, 'GAP/GAP_1.xml'))

    shutil.rmtree(os.path.join(example_dir, 'MD'))
    shutil.rmtree(os.path.join(example_dir, 'GAP'))
    os.remove(os.path.join(example_dir, 'errors.json'))
    os.remove('T')

if __name__ == '__main__':
    test_batch_gap_fit()
