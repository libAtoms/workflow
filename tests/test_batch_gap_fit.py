import os, shutil, sys

import numpy as np
import pytest


def test_batch_gap_fit():
    example_dir = os.path.join(
        os.path.dirname(__file__), '../', 'examples', 'iterative_gap_fit'
    )
    sys.path.append(example_dir)
    import batch_gap_fit

    batch_gap_fit.main(verbose=True)

    assert os.path.exists(os.path.join(example_dir, 'GAP'))
    assert os.path.exists(os.path.join(example_dir, 'MD'))
    assert os.path.exists(os.path.join(example_dir, 'errors.json'))
    assert os.path.exists(os.path.join(example_dir, 'GAP/GAP_5.xml'))

    shutil.rmtree(os.path.join(example_dir, 'MD'))
    shutil.rmtree(os.path.join(example_dir, 'GAP'))
    os.remove(os.path.join(example_dir, 'errors.json'))
    os.remove('T')

if __name__ == '__main__':
    test_batch_gap_fit()
