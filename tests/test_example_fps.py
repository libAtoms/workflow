import os, sys
from ase.io import read

def test_fps(n=10):
    example_dir = os.path.join(
            os.path.dirname(__file__), '../', 'examples', 'select_fps' 
            )
    sys.path.append(example_dir)
    import fps 
    fps.main(nsamples=n)

    assert os.path.exists(os.path.join(example_dir, 'md_desc.xyz'))
    assert os.path.exists(os.path.join(example_dir, 'out_fps.xyz'))
    
    fps = read(os.path.join(example_dir, "out_fps.xyz"), ":")
    assert len(fps) == n

    os.remove(os.path.join(example_dir, 'md_desc.xyz'))
    os.remove(os.path.join(example_dir, 'out_fps.xyz'))
       
if __name__ == '__main__':
    test_fps()
