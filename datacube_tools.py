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
    dc_persistance = np.zeros_like(dc)
    dc_persistance[:, :, 0] = dc[:, :, 0]
    for i in range(1, dc.shape[2]):
        dc_persistance[:, :, i] = func(dc[:, :, 0: i + 1], axis=axis)

    return dc_persistance
