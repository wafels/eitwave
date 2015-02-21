#
# Tools that implement mapcube operations 
#
import copy
import numpy as np
from sunpy.map.mapbase import GenericMap
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

    if not(isinstance(base, GenericMap)):
        base_map = mc[base]
    else:
        base_map = base

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


@mapcube_input
def accumulate(mc, accum, normalize=True):
    """
    Parameters
    ----------
    mc : sunpy.map.MapCube
       A sunpy mapcube object

    Returns
    -------
    sunpy.map.MapCube
       A summed mapcube in the map layer (time) direction.

    """

    # counter for number of maps.
    j = 0
    # storage for the returned maps
    maps = []
    nmaps = len(mc)
    while j + accum <= nmaps:
        i = 0
        while i < accum:
            thismap = mc[i + j]
            if normalize:
                normalization = thismap.exposure_time
            else:
                normalization = 1.0
            if i == 0:
                # Emission rate
                m = thismap.data / normalization
            else:
                # Emission rate
                m = m + thismap.data / normalization
            i = i + 1
        j = j + accum
        # Make a copy of the meta header and set the exposure time to accum,
        # indicating that 'n' normalized exposures were used.
        new_meta = copy.deepcopy(thismap.meta)
        new_meta['exptime'] = np.float64(accum)
        maps.append(Map(m, new_meta))
    # Create the new mapcube and return
    return Map(maps, cube=True)


@mapcube_input
def superpixel(mc, dimension, **kwargs):
    """
    Parameters
    ----------
    mc : sunpy.map.MapCube
       A sunpy mapcube object

    Returns
    -------
    sunpy.map.MapCube
       A mapcube containing maps that have had the map superpixel summing
       method applied to each layer.
    """

    # Storage for the returned maps
    maps = []
    for m in mc:
        maps.append(m.superpixel(dimension, **kwargs))
    # Create the new mapcube and return
    return Map(maps, cube=True)


@mapcube_input
def submap(mc, range_a, range_b, **kwargs):
    """
    Parameters
    ----------
    mc : sunpy.map.MapCube
       A sunpy mapcube object

    range_a : list


    range_b : list

    Returns
    -------
    sunpy.map.MapCube
       A mapcube containing maps that have had the map submap
       method applied to each layer.
    """
    nmc = len(mc)
    if (len(range_a) == nmc) and (len(range_b) == nmc):
        ra = range_a
        rb = range_b
    elif (len(range_a) == 1) and (len(range_b) == 1):
        ra = [range_a for i in range(0, nmc)]
        rb = [range_b for i in range(0, nmc)]
    else:
        raise ValueError('Both input ranges must be either of size 1 or size '
                         'equal to the number of maps in the mapcube')
        return None


    # Storage for the returned maps
    maps = []
    for im, m in enumerate(mc):
        maps.append(Map.submap(m, ra[im], rb[im], **kwargs))
    # Create the new mapcube and return
    return Map(maps, cube=True)
