#
# Tools that implement common datacube operations.  The first argument in each
# function is a numpy ndarray.  Datacubes are not the same as mapcubes.  
#
import numpy as np

# Decorator testing the input for these functions
def datacube_input(func):
    def check(*args, **kwargs):
        if not(isinstance(args[0], np.ndarray)):
            raise ValueError('First argument must be a numpy ndarray.')

        return func(*args, **kwargs)
    return check


@datacube_input
def persistence(dc, func=np.max, axis=2):
    """
    Take an input datacube and return the persistance cube.
    """
    newdc = np.zeros_like(dc)
    newdc[:, :, 0] = dc[:, :, 0]
    for i in range(1, dc.shape[2]):
        newdc[:, :, i] = func(dc[:, :, 0: i + 1], axis=axis)
    return newdc


@datacube_input
def running_difference(dc, offset=1):
    """
    Take the running difference of the input datacube
    """
    newdc = np.zeros_like(dc)
    for i in range(0, dc.shape[2] - offset):
        newdc[:, :, i] = dc[:, :, i + offset] - dc[:, :, i]
    return newdc
