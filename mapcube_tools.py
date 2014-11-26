#
# Tools that implement mapcube operations 
#
import numpy as np
from sunpy.map import Map
from sunpy.map import MapCube
from datacube_tools import persistence as persistence_dc


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
        newdata = mc[i + offset].data - mc[i].data
        if use_offset_for_meta:
            newmeta = mc[i + offset].meta
        else:
            newmeta = mc[i].meta
        newmc.append(Map(newdata, newmeta))

    # Create the new mapcube and return
    return Map(newmc, cube=True)


@mapcube_input
def base_difference(mc, base=0, fraction=False):
    """
    Calculate the base difference of a mapcube.

    Parameters
    ----------
    mc : sunpy.map.MapCube
       A sunpy mapcube object

    base : int, sunpy.map.Map
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
    sunpy.map.MapCube
       A mapcube containing base difference of the input mapcube.
    
    """

    if not(isinstance(base, sunpy.map.Map)):
        base_map = mc[base]

    if base_map.shape != mc[0].data.shape:
        raise ValueError('Base map does not have the same shape as the maps in the input mapcube.')

    # Fractional changes or absolute changes
    if fraction:
        relative = base_map.data
    else:
        relative = 1.0

    # Create a list containing the data for the new map object
    newmc = []
    for m in mc:
        newdata = (m.data - base_map.data) / relative
        newmc.append(Map(newdata, m.meta))

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
