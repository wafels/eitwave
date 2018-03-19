#
# Tools that implement mapcube operations
#
import os
from copy import deepcopy
import datetime

import numpy as np
from numpy.random import poisson

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.visualization import LinearStretch, PercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize

from sunpy.map.mapbase import GenericMap
from sunpy.map import Map
from sunpy.map import MapCube
from sunpy.time import parse_time

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


# Given a list of times, get the max time
def _max_time(time_list):
    return max([parse_time(d) for d in time_list])


@mapcube_input
def calculate_movie_normalization(mc, percentile_interval=99.0, stretch=None):
    """
    A convenience function that calculates an image normalization
    that means a movie of the input mapcube will not flicker.
    Assumes that all layers are similar and the stretch function
    for all layers is the same

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
    An image normalization setting that can be used with all the images
    in the mapcube ensuring no flickering in a movie of the images.
    """
    vmin, vmax = PercentileInterval(percentile_interval).get_limits(mc.as_array())
    if stretch is None:
        try:
            stretcher = mc[0].plot_settings['norm'].stretch
        except AttributeError:
            stretcher = None
    else:
        stretcher = stretch
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretcher)


@mapcube_input
def apply_movie_normalization(mc, image_normalization):
    """
    A convenience function that applies an image normalization
    that means a movie of the input mapcube will not flicker.
    Assumes that all layers are similar and the stretch function
    for all layers is the same

    Parameters
    ----------
    mc : `sunpy.map.MapCube`
        a sunpy mapcube

    normaliztion : `~astropy.visualization.ImageNormalize`
        image stretch function

    Returns
    -------
    An image normalization setting that can be used with all the images
    in the mapcube ensuring no flickering in a movie of the images.
    """
    new_mc = []
    for m in mc:
        m_new = deepcopy(m)
        m_new.plot_settings["norm"] = image_normalization
        new_mc.append(m_new)
    return Map(new_mc, cube=True)


@mapcube_input
def running_difference(mc, offset=1, use_offset_for_meta='ahead'):
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
            plot_settings = mc[i + offset].plot_settings
        elif use_offset_for_meta == 'behind':
            new_meta = mc[i].meta
            plot_settings = mc[i].plot_settings
        elif use_offset_for_meta == 'mean':
            new_meta = deepcopy(mc[i + offset].meta)
            new_meta['date_obs'] = _mean_time([parse_time(mc[i + offset].date),
                                               parse_time(mc[i].date)])
            plot_settings = mc[i + offset].plot_settings
        else:
            raise ValueError('The value of the keyword "use_offset_for_meta" has not been recognized.')

        # Update the plot scaling.  The default here attempts to produce decent
        # looking images
        new_map = Map(new_data, new_meta)
        new_map.plot_settings = plot_settings
        new_mc.append(new_map)

    # Create the new mapcube and return
    return Map(new_mc, cube=True)


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
    return Map(new_mc, cube=True)


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
       A mapcube containing the persistence transform of the input mapcube.
       The value normalization function used in plotting the data is changed,
       prettifying movies of resultant mapcube.
    """

    # Get the persistence transform
    new_datacube = persistence_dc(mc.as_array(), func=func)

    # Create a list containing the data for the new map object
    new_mc = []
    for i, m in enumerate(mc):
        new_map = Map(new_datacube[:, :, i], m.meta)
        new_map.plot_settings = deepcopy(m.plot_settings)
        new_mc.append(new_map)

    # Create the new mapcube and return
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
        new_meta['date_obs'] = _max_time(these_map_times)

        # Create the map list that will be used to make the mapcube
        new_map = Map(m, new_meta)
        new_map.plot_settings = deepcopy(this_map.plot_settings)
        maps.append(new_map)

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


@mapcube_input
def write_layers(mc, directory, prefix, filetype='png', **savefig_kwargs):
    """
    Write a numerically ordered set of images out from a mapcube.  This
    function is useful for making movies using FFMPEG.

    :param mc:
    :param directory:
    :param prefix:
    :param filetype:
    :return:
    """
    # Numbering width
    width = 1 + int(np.ceil(np.log10(len(mc))))

    # Write an image for each map
    for i, m in enumerate(mc):
        plt.close('all')
        file_number = '{0:0{width}}'.format(i, width=width)
        filepath = os.path.join(os.path.expanduser(directory), '{:s}_{:s}.{:s}'.format(prefix, file_number, filetype))
        ret = m.plot()
        m.draw_grid(color='blue')
        plt.grid('on', color='cyan', linestyle=":")
        plt.savefig(filepath, **savefig_kwargs)

    # Optionally make a movie
    #if make_movie:
    #    cmd = 'avconv -framerate 25 -f image2 -i emission_longetal2014_figure4_%4d.png -c:v h264 -crf 1 out.mov'


def data_simple_replace_zero_values(data, replacement_value=0.001):
    """
    Replace zero values in a numpy array with a fixed replacement value.

    :param data:
    :param replacement_value:
    :return:
    """
    return data_simple_replace(data, data == 0, replacement_value)


def data_simple_replace_negative_values(data, replacement_value=0.001):
    """
    Replace negative values in a numpy array with a fixed replacement value.

    :param data:
    :param replacement_value:
    :return:
    """
    return data_simple_replace(data, data < 0, replacement_value)


def data_simple_replace_nans(data, replacement_value=0.001):
    """
    Replace NaNs in a numpy array with a fixed replacement value.

    :param data:
    :param replacement_value:
    :return:
    """
    return data_simple_replace(data, ~np.isfinite(data), replacement_value)


def data_simple_replace(data, condition, replacement_value):
    """
    Replace values in a numpy array with the replacement value where the input
    condition is True and return a new array.

    :param data:
    :param condition:
    :param replacement_value:
    :return:
    """
    newdata = deepcopy(data)
    newdata[condition] = replacement_value
    return newdata


def map_simple_replace(smap, condition, replacement_value):
    newdata = deepcopy(smap.data)
    newdata[condition] = replacement_value
    return Map(newdata, smap.meta)


def map_noisy_realization(smap):
    """
    Return a Poisson-noisy version of the input map.

    :param smap:
    :return:
    """
    return Map(poisson(lam=smap.data), smap.meta)


def mapcube_noisy_realization(mc,
                              simple_replace_nans=True,
                              simple_replace_negative_values=True,
                              simple_replace_zero_values=True):
    """
    Return a Poisson noisy version of the input mapcube.

    :param mc:
    :param simple_replace_nans:
    :param simple_replace_negative_values:
    :param simple_replace_zero_values:
    :return:
    """
    # Get all the data from the mapcube
    data = mc.as_array()

    # Apply the simple replacements as necessary
    if simple_replace_nans:
        data = data_simple_replace_nans(data)
    if simple_replace_negative_values:
        data = data_simple_replace_negative_values(data)
    if simple_replace_zero_values:
        data = data_simple_replace_zero_values(data)

    # Create the noisy mapcube and return it
    noisy_mapcube = []
    for i, m in enumerate(mc):
        noisy_mapcube.append(map_noisy_realization(Map(data[:, :, i], m.meta)))
    return Map(noisy_mapcube, cube=True)
