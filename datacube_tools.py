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
def running_difference(dc, offset=1):
    """
    Take the running difference of the input datacube
    """
    newdc = np.zeros_like(dc)
    for i in range(0, dc.shape[2] - offset):
        newdc[:, :, i] = dc[:, :, i + offset] - dc[:, :, i]
    return newdc


#
# base difference
#
@datacube_input
def base_difference(dc, base=0, fraction=False):
    """
    Calculate the base difference of a datacube.

    Parameters
    ----------
    dc : three dimensional numpy array where the first two dimensions are
         space.

    base : int, two-dimensional numpy array
       If base is an integer, this is understood as an index to the input
       mapcube.  Differences are calculated relative to the map at index
       'base'.  If base is a sunpy map, then differences are calculated
       relative to that map

    fraction : boolean
        If False, then absolute changes relative to the base map are
        returned.  If True, then fractional changes relative to the base map
        are returned

    Returns
    -------
    numpy 3 dimensional array
       A data cube containing base difference of the input data cube.

    """

    if not(isinstance(base, np.ndarray)):
        base_data = dc[:, :, 0]
    else:
        base_data = base

    if base_data.shape != dc.shape[0:2]:
        raise ValueError('Base map does not have the same shape as the maps in the input datacube.')

    # Fractional changes or absolute changes
    if fraction:
        relative = base_data
    else:
        relative = 1.0

    # Create a list containing the data for the new map object
    new_datacube = np.zeros_like(dc)
    for i in range(0, dc.shape[2]):
        new_datacube[:, :, i] = (dc[:, :, i] - base_data) / relative

    return new_datacube


#
# Running difference
#
@datacube_input
def running_difference(dc, offset=1):
    """
    Calculate the running difference of a datacube.

    Parameters
    ----------
    dc : three dimensional numpy array where the first two dimensions are
         space.

    offset : [ int ]
       Calculate the running difference between map 'i + offset' and image 'i'.

    Returns
    -------
    three dimensional numpy array
       A datacube containing the running difference of the input datacube.

    """
    ny = dc.shape[0]
    nx = dc.shape[1]
    nt = dc.shape[2]
    new_datacube = np.zeros((ny, nx, nt-offset))
    for i in range(0, nt-offset):
        new_datacube[:, :, i] = dc[:, :, i + offset] - dc[:, :, i]

    return new_datacube


@datacube_input
def persistence(dc, func=np.max, axis=2):
    """
    Take an input datacube and return the persistence cube.
    """
    newdc = np.zeros_like(dc)
    newdc[:, :, 0] = dc[:, :, 0]
    for i in range(1, dc.shape[2]):
        newdc[:, :, i] = func(dc[:, :, 0: i + 1], axis=axis)
    return newdc