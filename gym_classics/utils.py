import numpy as np

def clip(x, low, high):
    """A scalar version of numpy.clip. Much faster because it avoids memory allocation."""
    return min(max(x, low), high)


# np.argmax does not break ties randomly
def random_argmax(x, axis = None):
    """
    Argmax that breaks ties randomly. If axis is None, returns a single index. 
    If axis is specified, returns an array of indices along that axis.
    """
    if axis is None:
        return np.random.choice(np.where(x == np.max(x))[0])
    else:
        return np.apply_along_axis(random_argmax, axis, x)
