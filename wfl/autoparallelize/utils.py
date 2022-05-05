import itertools


def grouper(n, iterable):
    """iterator that goes over iterable in specified size groups

    Parameters
    ----------
    iterable: any iterable
        iterable to loop over
    n: int
        size of group in each returned tuple

    Returns
    -------
    sequence of tuples, with items from iterable, each of size n (or smaller if n items are not available)
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
