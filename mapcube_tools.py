#
# Tools that implement mapcube operations
#
from copy import deepcopy
import datetime
import numpy as np
import astropy.units as u
from astropy.visualization import LinearStretch, PercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize

from sunpy.map.mapbase import GenericMap
from sunpy.map import Map
from sunpy.map import MapCube
from sunpy.time import parse_time


# Decorator testing the input for these functions
def mapcube_input(func):
    def check(*args, **kwargs):
        if not(isinstance(args[0], MapCube)):
            raise ValueError('First argument must be a sunpy MapCube.')

        if not(args[0].all_maps_same_shape()):
            raise ValueError('All maps in the input mapcube must have the same shape.')

        return func(*args, **kwargs)
    return check


def persistence_dc(dc, func=np.max, axis=2):
    """
    Take an input datacube and return the persistence cube.
    """
    newdc = np.zeros_like(dc)
    newdc[:, :, 0] = dc[:, :, 0]
    for i in range(1, dc.shape[2]):
        newdc[:, :, i] = func(dc[:, :, 0: i + 1], axis=axis)
    return newdc


# Get the relative changes in times in comparison to a base time
def _time_deltas(base_time, time_list):
    return [(t - base_time).total_seconds() for t in time_list] * u.s


# Average change in times
def _average_time_delta_in_seconds(base_time, time_list):
    return np.mean(_time_deltas(base_time, time_list).to('s').value) * u.s


# Given a list of times, get the mean time
def _mean_time(time_list):
    base_time = time_list[0]
    delta_t = _average_time_delta_in_seconds(base_time, time_list)
    return base_time + datetime.timedelta(seconds=delta_t.value)


@mapcube_input
def movie_normalization(mc, percentile_interval=99.0, stretch=None):
    """
    Return a mapcube such that each map in the mapcube has the same variable
    limits.  If each map also has the same stretch function, then movies of
    the mapcube will not flicker.

    Parameters
    ----------
    mc : `sunpy.map.MapCube`
        a sunpy mapcube

    percentile_interval : float
        the central percentile interval used to

    stretch :
        image stretch function

    Returns
    -------
    The input mapcube is returned with the same variable limits on the image
    normalization for each map in the mapcube.
    """
    vmin, vmax = PercentileInterval(percentile_interval).get_limits(mc.as_array())
    for i, m in enumerate(mc):
        if stretch is None:
            stretcher = m.plot_settings['norm'].stretch
        else:
            stretcher = stretch
        mc[i].plot_settings['norm'] = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretcher)
    return mc


@mapcube_input
def running_difference(mc, offset=1, use_offset_for_meta='mean',
                       image_normalize=True):
    """
    Calculate the running difference of a mapcube.

    Parameters
    ----------
    mc : sunpy.map.MapCube
       A sunpy mapcube object

    offset : [ int ]
       Calculate the running difference between map 'i + offset' and image 'i'.

    use_offset_for_meta : {'ahead', 'behind', 'mean'}
       Which meta header to use in layer 'i' in the returned mapcube, either
       from map 'i + offset' (when set to 'ahead') and image 'i' (when set to
       'behind').  When set to 'mean', the ahead meta object is copied, with
       the observation date replaced with the mean of the ahead and behind
       observation dates.

    image_normalize : bool
        If true, return the mapcube with the same image normalization applied
        to all maps in the mapcube.

    Returns
    -------
    sunpy.map.MapCube
       A mapcube containing the running difference of the input mapcube.
       The value normalization function used in plotting the data is changed,
       prettifying movies of resultant mapcube.
    """
    # Create a list containing the data for the new map object
    new_mc = []
    for i in range(0, len(mc.maps) - offset):
        new_data = mc[i + offset].data - mc[i].data
        if use_offset_for_meta == 'ahead':
            new_meta = mc[i + offset].meta
        elif use_offset_for_meta == 'behind':
            new_meta = mc[i].meta
        elif use_offset_for_meta == 'mean':
            new_meta = deepcopy(mc[i + offset].meta)
            new_meta['date_obs'] = _mean_time([parse_time(mc[i + offset].date),
                                               parse_time(mc[i].date)])
        else:
            raise ValueError('The value of the keyword "use_offset_for_meta" has not been recognized.')

        # Update the plot scaling.  The default here attempts to produce decent
        # looking images
        new_mc.append(Map(new_data, new_meta))

    # Create the new mapcube and return
    if image_normalize:
        return movie_normalization(Map(new_mc, cube=True), stretch=LinearStretch())
    else:
        return Map(new_mc, cube=True)


@mapcube_input
def base_difference(mc, base=0, fraction=False, image_normalize=True):
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

    image_normalize : bool
        If true, return the mapcube with the same image normalization applied
        to all maps in the mapcube.

    Returns
    -------
    sunpy.map.MapCube
       A mapcube containing base difference of the input mapcube.
       The value normalization function used in plotting the data is changed,
       prettifying movies of resultant mapcube.
    """

    if not(isinstance(base, GenericMap)):
        base_data = mc[base].data
    else:
        base_data = base.data

    if base_data.shape != mc[0].data.shape:
        raise ValueError('Base map does not have the same shape as the maps in the input mapcube.')

    # Fractional changes or absolute changes
    if fraction:
        relative = base_data
    else:
        relative = 1.0

    # Create a list containing the data for the new map object
    new_mc = []
    for m in mc:
        new_data = (m.data - base_data) / relative
    new_mc.append(Map(new_data, m.meta))

    # Create the new mapcube and return
    if image_normalize:
        return movie_normalization(Map(new_mc, cube=True), stretch=LinearStretch())
    else:
        return Map(new_mc, cube=True)


@mapcube_input
def persistence(mc, func=np.max, image_normalize=True):
    """
    Parameters
    ----------
    mc : sunpy.map.MapCube
       A sunpy mapcube object

    Returns
    -------
    sunpy.map.MapCube
       A mapcube containing the persistence transform of the input mapcube.
       The value normalization function used in plotting the data is changed,
       prettifying movies of resultant mapcube.
    """

    # Get the persistence transform
    new_datacube = persistence_dc(mc.as_array(), func=func)

    # Create a list containing the data for the new map object
    new_mc = []
    for i, m in enumerate(mc):
        new_mc.append(Map(new_datacube[:, :, i], m.meta))

    # Create the new mapcube and return
    if image_normalize:
        return movie_normalization(Map(new_mc, cube=True))
    else:
        return Map(new_mc, cube=True)


@mapcube_input
def accumulate(mc, accum, normalize=True):
    """
    Parameters
    ----------
    mc : sunpy.map.MapCube
       A sunpy mapcube object

    accum :

    normalize :

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
        these_map_times = []
        while i < accum:
            this_map = mc[i + j]
            these_map_times.append(parse_time(this_map.date))
            if normalize:
                normalization = this_map.exposure_time
            else:
                normalization = 1.0
            if i == 0:
                # Emission rate
                m = this_map.data / normalization
            else:
                # Emission rate
                m += this_map.data / normalization
            i += 1
        j += accum
        # Make a copy of the meta header and set the exposure time to accum,
        # indicating that 'n' normalized exposures were used.
        new_meta = deepcopy(this_map.meta)
        new_meta['exptime'] = np.float64(accum)

        # Set the observation time to the average of the times used to form
        # the map.
        new_meta['date_obs'] = _mean_time(these_map_times)

        # Create the map list that will be used to make the mapcube
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

    # Storage for the returned maps
    maps = []
    for im, m in enumerate(mc):
        maps.append(Map.submap(m, ra[im], rb[im], **kwargs))
    # Create the new mapcube and return
    return Map(maps, cube=True)


@mapcube_input
def multiply(mc1, mc2, use_meta=1):
    """
    Multiply the data values in the input map cubes and return
    a new mapcube.

    :param mc1:
    :param mc2:
    :param use_meta:
    :return:
    """
    if len(mc1) != len(mc2):
        raise ValueError('Input mapcubes have different number of maps.')
    new_mc = []
    nt = len(mc1)
    for i in range(0, nt):
        new_data = np.multiply(mc1[i].data, mc2[i].data)
        if use_meta == 1:
            new_mc.append(Map(new_data, mc1[i].meta))
        elif use_meta == 2:
            new_mc.append(Map(new_data, mc2[i].meta))
        else:
            raise ValueError('The use_meta keyword needs the value 1 or 2.')
    return Map(new_mc, cube=True)
