#
# Tools that implement mapcube operations 
#
import numpy as np
from sunpy.map import Map
from sunpy.map import MapCube
from datacube_tools import persistence_dc


# Decorator testing the input for these functions
def mapcube_input(func):
    def check(*args, **kwargs):
        if not(isinstance(args[0], MapCube)):
            raise ValueError('First argument must be a sunpy MapCube.')
    
        if not(args[0].all_maps_same_shape()):
            raise ValueError('All maps in the input mapcube must have the same shape.')

        return func(*args, **kwargs)
    return check


@mapcube_input
def data(mc):
    """
    Take an input mapcube and return a three-dimensional numpy array - a
    datacube.  Chickens go in, pies come out.  Only needed until the mapcube.as_array() function
    is a part of the sunpy release

    Parameters
    ----------
    mc : sunpy.map.MapCube
       A sunpy mapcube object

    Returns
    -------
    ndarray
       A numpy ndarray of dimension (ny, nx, nt).
    """
    nt = len(mc.maps)
    shape = mc[0].data.shape
    ny = shape[0]
    nx = shape[1]
    dc = np.zeros((ny, nx, nt))
    for i in range(0, nt):
        dc[:, :, i] = mc[i].data
    return dc


@mapcube_input
def running_difference(mc, offset=1, use_offset_for_meta=True):
    """
    Calculate the running difference of a mapcube.

    Parameters
    ----------
    mc : sunpy.map.MapCube
       A sunpy mapcube object

    offset : [ int ]
       Calculate the running difference between map 'i + offset' and image 'i'.
    use_offset_for_meta : boolean
       Which meta header to use in layer 'i' in the returned mapcube, either
       from map 'i + offset' (when set to True) and image 'i' (when set to
       False).

    Returns
    -------
    sunpy.map.MapCube
       A mapcube containing the running difference of the input mapcube.
    
    """

    # Create a list containing the data for the new map object
    newmc = []
    for i in range(0, len(mc.maps) - offset):
        newdata = mc.maps.data[i + offset] - mc.maps.data[i]
        if use_offset_for_meta:
            newmeta = mc.maps.meta[i + offset]
        else:
            newmeta = mc.maps.meta[i]
        newmc.append(Map(newdata, newmeta))

    # Create the new mapcube and return
    return Map(newmc, cube=True)


@mapcube_input
def persistence(mc, func=np.max):
    """
    Parameters
    ----------
    mc : sunpy.map.MapCube
       A sunpy mapcube object

    Returns
    -------
    sunpy.map.MapCube
       A mapcube containing the running difference of the input mapcube.

    """

    # Get the persistence
    persistence_cube = persistence_dc(mc.as_array(), func=func)

    # Create a list containing the data for the new map object
    newmc = []
    for i in range(0, len(mc.maps)):
        newmc.append(Map(persistence_cube[:, :, i], mc[i].meta))

    # Create the new mapcube and return
    return Map(newmc, cube=True)
