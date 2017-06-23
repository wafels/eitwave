#
# Utility functions for AWARE
#
import re
import os
from copy import deepcopy
import pickle

import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.time import TimeRange, parse_time
from sunpy.map import Map
from sunpy.coordinates import frames

import aware_get_data
import aware_constants

eitwave_data_root = aware_constants.eitwave_data_root
solar_circumference_per_degree_in_km = aware_constants.solar_circumference_per_degree_in_km


def convert_dict_to_single_string(d, sep='__'):
    """
    Converts a dictionary into a single string with key-value pairs separated
    by the separator

    :param d:
    :param sep:
    :return:
    """
    pairs = [str(key)+'='+str(d[key]) for key in list(d.keys())]
    return sep.join(pair for pair in pairs)


def create_input_to_aware_for_test_observational_data(wave_name,

                                                      instrument='AIA',
                                                      wavelength=211,
                                                      event_type='FL',
                                                      root_directory=eitwave_data_root):
    # Set up the data
    if wave_name == 'longetal2014_figure4':
        hek_record_index = 0

    if wave_name == 'longetal2014_figure7':
        hek_record_index = 0

    if wave_name == 'longetal2014_figure8a':
        hek_record_index = 0
        time_range = ['2011-02-15 01:48:00', '2011-02-15 02:14:34']

    if wave_name == 'longetal2014_figure8e':
        hek_record_index = 0
        time_range = ['2011-02-16 14:22:36', '2011-02-16 14:39:48']

    if wave_name == 'byrneetal2013_figure12':
        hek_record_index = 0
        time_range = ['2010-08-14 09:40:18', '2010-08-14 10:32:00']

    # Where the data is stored
    wave_info_location = os.path.join(root_directory, wave_name)

    # Get the FITS files we are interested in
    fits_location = os.path.join(wave_info_location, instrument, str(wavelength), 'fits', '1.0')
    fits_file_list = aware_get_data.get_file_list(fits_location, '.fits')
    if len(fits_file_list) == 0:
        instrument_measurement, qr = aware_get_data.find_fits(time_range, instrument, wavelength)
        print('Downloading {:n} files'.format(len(qr)))
        fits_file_list = aware_get_data.download_fits(qr, instrument_measurement=instrument_measurement)

    # Get the source information
    source_location = os.path.join(wave_info_location, event_type)
    source_path = aware_get_data.get_file_list(source_location, '.pkl')
    if len(source_path) == 0:
        print('Querying HEK for trigger data.')
        hek_record = aware_get_data.download_trigger_events(time_range)
    else:
        f = open(source_path[0], 'rb')
        hek_record = pickle.load(f)
        f.close()

    analysis_time_range = TimeRange(hek_record[hek_record_index]['event_starttime'],
                                    time_from_file_name(fits_file_list[-1].split(os.path.sep)[-1]))

    for_analysis = []
    for f in fits_file_list:
        g = f.split(os.path.sep)[-1]
        if (time_from_file_name(g) <= analysis_time_range.end) and (time_from_file_name(g) >= analysis_time_range.start):
            for_analysis.append(f)

    return {'finalmaps': aware_get_data.accumulate_from_file_list(for_analysis),
            'epi_lat': hek_record[hek_record_index]['hgs_y'],
            'epi_lon': hek_record[hek_record_index]['hgs_x']}


def time_from_file_name(f, fits_level=1.0):
    if fits_level == 1.0:
        start = 14

    year = f[start:start+4]
    month = f[start+5:start+7]
    day = f[start+8:start+10]
    hour = f[start+11:start+13]
    minute = f[start+14:start+16]
    second = f[start+17:start+19]

    time = '{:s}-{:s}-{:s} {:s}:{:s}:{:s}'.format(year, month, day, hour, minute, second)

    return parse_time(time)


###############################################################################
#
# AWARE arc and wave measurement scores
#
def arc_duration_fraction(defined, nt):
    """
    :param defined: boolean array of where there is a detection
    :param nt: number of possible detections
    :return: fraction of the full duration that the detection exists
    """
    return np.float64(np.sum(defined)) / np.float64(nt)


#
# Long et al (2014) score function
#
class ScoreLong:
    """
    Calculate the Long et al (2014) score function.
    """
    def __init__(self, velocity, acceleration, sigma_d, d, nt, indicesf,
                 velocity_range=[1, 2000] * u.km/u.s,
                 acceleration_range=[-2.0, 2.0] * u.km/u.s/u.s,
                 sigma_rel_limit=0.5,
                 dynamic_component_weight=0.5,
                 use_maximum_measureable_extent=True):
        self.velocity = velocity * solar_circumference_per_degree_in_km
        if acceleration is not None:
            self.acceleration = acceleration * solar_circumference_per_degree_in_km
        else:
            self.acceleration = None
        self.sigma_d = sigma_d
        self.d = d
        self.nt = nt
        self.indicesf = indicesf
        self.velocity_range = velocity_range
        self.acceleration_range = acceleration_range
        self.sigma_rel_limit = sigma_rel_limit
        self.dynamic_component_weight = dynamic_component_weight
        self.use_maximum_measureable_extent = use_maximum_measureable_extent

        # Velocity fit - is it acceptable?
        if (self.velocity > self.velocity_range[0]) and (self.velocity < self.velocity_range[1]):
            self.velocity_score = 1.0
        else:
            self.velocity_score = 0.0
        self.velocity_is_dynamic_component = 1.0

        # Acceleration fit - is it acceptable?
        if self.acceleration is not None:
            self.acceleration_is_dynamic_component = 1.0
            if (self.acceleration > self.acceleration_range[0]) and (self.acceleration < self.acceleration_range[1]):
                self.acceleration_score = 1.0
            else:
                self.acceleration_score = 0.0
        else:
            self.acceleration_is_dynamic_component = 0.0
            self.acceleration_score = 0.0

        # Did the fit along the arc have a reasonable errors on average?
        self.sigma_rel = np.mean(sigma_d/d)
        if self.sigma_rel < self.sigma_rel_limit:
            self.sigma_rel_score = 1.0
        else:
            self.sigma_rel_score = 0.0
        self.sigma_is_dynamic_component = 1.0

        # Final dynamic component of the score
        self.n_dynamic_components = self.velocity_is_dynamic_component + \
                                    self.acceleration_is_dynamic_component + \
                                    self.sigma_is_dynamic_component
        self.dynamic_component = self.dynamic_component_weight*(self.velocity_score +
                                                            self.acceleration_score +
                                                            self.sigma_rel_score) / self.n_dynamic_components

        # Which time to use to assess the existence time
        # Can use the number of measurements made, or use
        # the maximum extent from the first to the last
        # measureable times
        if not self.use_maximum_measureable_extent:
            self.existence_component_time = self.nt
        else:
            self.existence_component_time = 1 + self.indicesf[-1] - self.indicesf[0]

        # Existence component - how much of the data along the arc was fit?
        self.existence_component = (1-self.dynamic_component_weight) * len(self.d) / (1.0 * self.existence_component_time)

        # Return the score in the range 0-100
        self.final_score = 100*(self.existence_component + self.dynamic_component)


# AWARE wave assessment
def assess_wave(results):
    """
    Takes an AWARE results dictionary and calculates an assessment of the
    properties of the wave

    :param results:
    :return:
    """
    theta_wave = None
    n_segments = None
    median_long_score = None
    return theta_wave, n_segments, median_long_score


###############################################################################
#
# AWARE - make a plot of the progress of the detected wave front.
#
def wave_progress_map_by_running_difference(mc, level=0.0, index=0):
    """
    Take an input mapcube and return the location of the wavefront as a
    function of time in a single SunPy map.  The locations are color-coded,
    where the color indicates time.  Locations are calculated by first
    thresholding the data, creating a mask, subtracting running difference
    masks to find where the wavefront went to, and then adding those locations
    to the map.

    mc : sunpy.map.MapCube
        Input mapcube

    index : int
        Index of the map in the input mapcube that

    Return
    ------
    map, list

    A tuple containing the following: a sunpy map with values in the range 1 to
    len(input mapcube) where larger numbers indicate later times, and a list
    that holds the corresponding timestamps.  If a pixel in the map has value
    'n', then the wavefront is located at time timestamps[n].
    """

    wave_progress_data = np.zeros_like(mc[index].data)
    timestamps = []
    for im in range(0, len(mc)-1):
        data1 = mc[im+1].data
        detection1 = data1 > level

        data0 = mc[im].data
        detection0 = data0 > level

        new_detection = detection1.astype(int) - detection0.astype(int)

        progress_index = new_detection > 0
        wave_progress_data[progress_index] = im + 1

        # Keep a record of the timestamps
        timestamps.append(mc[im+1].date)

    return Map(wave_progress_data, mc[index].meta), timestamps


def wave_progress_map_by_location(mc, level=0.0, index=0, minimum_value=0):
    """
    Take an input mapcube and return the location of the wavefront as a
    function of time in a single SunPy map.  The locations are color-coded,
    where the color indicates time.  Locations are calculated by first
    thresholding the data and creating a mask.  That mask is placed in the
    results map.

    mc : sunpy.map.MapCube
        Input mapcube

    index : int
        Index of the map in the input mapcube that

    Return
    ------
    map, list

    A tuple containing the following: a sunpy map with values in the range 1 to
    len(input mapcube) where larger numbers indicate later times, and a list
    that holds the corresponding timestamps.  If a pixel in the map has value
    'n', then the wavefront is located at time timestamps[n].
    """

    wave_progress_data = np.zeros_like(mc[index].data)
    timestamps = []
    for im in range(0, len(mc)):
        detection = mc[im].data > level
        wave_progress_data[detection] = im + minimum_value

        # Keep a record of the timestamps
        timestamps.append(mc[im].date)

    return Map(wave_progress_data, mc[index].meta), timestamps


###############################################################################
#
# AWARE - make a plot of the progress of the detected wave front.
#
def progress_mask(mc, lower_limit=0.0):
    """
    Take an input AWARE-processed detection and return a binary mask
    mapcube that shows where the data is.

    mc : sunpy.map.MapCube
        Input mapcube

    Return
    ------
    mapcube
    """

    pm = []
    for im, m in enumerate(mc):
        wave_location_mask = 1.0*(m.data > lower_limit)
        pm.append(Map(wave_location_mask, m.meta))

    return Map(pm, cube=True)


###############################################################################
#
# AWARE - make a map of the Long et al (2014) score.
#
def long_score_map(aware_results):
    """
    Take an AWARE_results structure and return a SunPy map that shows the Long
    et al (2014) score.

    Return
    ------
    map
    """
    pass


###############################################################################
#
# AWARE - make movie.
#

def draw_limb(fig, ax, sunpy_map):
    p = sunpy_map.draw_limb()
    return p


#
def write_movie(mc, filename):
    """
    Take a mapcube and produce a movie of it.

    :param mc:
    :param filename:
    :return:
    """
    ani = mc.plot(plot_function=draw_limb)
    Writer = animation.writers['avconv']
    writer = Writer(fps=10, metadata=dict(artist='SunPy'), bitrate=18000)
    ani.save('{:s}.mp4'.format(filename), writer=writer)
    plt.close('all')


# Test symmetry of wave
def test_symmetry_of_wave(mc, image_root='/home/ireland/eitwave/img/test_symmetry_of_wave/'):
    for i, m in enumerate(mc):
        data1 = deepcopy(m.data)
        data2 = deepcopy(data1)

        left_minus_right = data1[:, :] - data2[::-1, :]
        upper_minus_lower = data1[:, :] - data2[:, ::-1]

        plt.ioff()
        plt.imshow(left_minus_right, origin='lower')
        plt.title('left - right ({:n})'.format(i))
        filename = os.path.join(image_root, 'left_minus_right_{:03n}.png'.format(i))
        plt.colorbar()
        plt.savefig(filename)
        plt.close('all')

        plt.ioff()
        plt.imshow(upper_minus_lower, origin='lower')
        plt.title('upper - lower ({:n})'.format(i))
        filename = os.path.join(image_root, 'upper_minus_lower_{:03n}.png'.format(i))
        plt.colorbar()
        plt.savefig(filename)
        plt.close('all')


def clean_for_overleaf(s, rule='\W+', rep='_'):
    return re.sub(rule, rep, s)


def great_arc(start, end, center=None, number_points=100):
    """
    Calculate a user-specified number of points on a great arc between a start
    and end point on a sphere.
    Parameters
    ----------
    start : `~astropy.coordinates.SkyCoord`
        Start point.
    end : `~astropy.coordinates.SkyCoord`
        End point.
    center : `~astropy.coordinates.SkyCoord`
        Center of the sphere.
    number_points : int
        Number of points along the great arc.
    Returns
    -------
    arc : `~astropy.coordinates.SkyCoord`
        Co-ordinates along the great arc in the co-ordinate frame of the
        start point.
    Example
    -------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> import sunpy.coordinates
    >>> from sunpy.coordinates import great_arc
    >>> import sunpy.map
    >>> from sunpy.data.sample import AIA_171_IMAGE
    >>> m = sunpy.map.Map(AIA_171_IMAGE)
    >>> a = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=m.coordinate_frame)
    >>> b = SkyCoord(-100*u.arcsec, 800*u.arcsec, frame=m.coordinate_frame)
    >>> v = great_arc(a, b)
    """

    # Create a helper object that contains all the information we need.
    gc = HelperGreatArcConvertToCartesian(start, end, center)

    # Calculate the points along the great arc.
    great_arc_points_cartesian = calculate_great_arc(gc.start_cartesian, gc.end_cartesian, gc.center_cartesian, number_points)*gc.start_unit

    # Transform the great arc back into the input frame.
    return SkyCoord(great_arc_points_cartesian[:, 0],
                    great_arc_points_cartesian[:, 1],
                    great_arc_points_cartesian[:, 2],
                    frame=frames.Heliocentric, observer=gc.observer).transform_to(gc.start_frame)


def great_arc_distance(start, end, center=None):
    """
    Calculate the distance between the start point and end point on a sphere.
    Parameters
    ----------
    start : `~astropy.coordinates.SkyCoord`
        Start point.
    end : `~astropy.coordinates.SkyCoord`
        End point.
    center : `~astropy.coordinates.SkyCoord`
        Center of the sphere.
    Returns
    -------
    distance : `astropy.units.Quantity`
        The distance between the two points on the sphere.
    """
    # Create a helper object that contains all the information we needed.
    gc = HelperGreatArcConvertToCartesian(start, end, center)

    # Calculate the properties of the great arc.
    this_great_arc = GreatArcPropertiesCartesian(gc.start_cartesian, gc.end_cartesian, gc.center_cartesian)

    # Return the distance on the sphere in the Cartesian distance units.
    return this_great_arc.distance * gc.start_unit


def great_arc_angular_separation(start, end, center=None):
    """
    Calculate the angular separation between the start point and end point on a
    sphere.
    Parameters
    ----------
    start : `~astropy.coordinates.SkyCoord`
        Start point.
    end : `~astropy.coordinates.SkyCoord`
        End point.
    center : `~astropy.coordinates.SkyCoord`
        Center of the sphere.
    Returns
    -------
    separation : `astropy.units.Quantity`
        The angular separation between the two points on the sphere.
    """
    # Create a helper object that contains all the information needed.
    gc = HelperGreatArcConvertToCartesian(start, end, center)

    # Calculate the properties of the great arc
    this_great_arc = GreatArcPropertiesCartesian(gc.start_cartesian, gc.end_cartesian, gc.center_cartesian)

    # Return the angular separation on the sphere.
    return np.rad2deg(this_great_arc.inner_angle) * u.degree


def calculate_great_arc(start_cartesian, end_cartesian, center_cartesian, number_points):
    """
    Calculate a user-specified number of points on a great arc between a start
    and end point on a sphere where the start and end points are assumed to be
    x,y,z Cartesian triples on a sphere relative to a center.
    Parameters
    ----------
    start_cartesian : `~numpy.ndarray`
        Start point expressed as a Cartesian xyz triple.
    end_cartesian : `~numpy.ndarray`
        End point expressed as a Cartesian xyz triple.
    center_cartesian : `~numpy.ndarray`
        Center of the sphere expressed as a Cartesian xyz triple
    number_points : int
        Number of points along the great arc.
    Returns
    -------
    arc : `~numpy.ndarray`
        Co-ordinates along the great arc expressed as Cartesian xyz triples.
        The shape of the array is (num, 3).
    """
    this_great_arc = GreatArcPropertiesCartesian(start_cartesian, end_cartesian, center_cartesian)

    # Range through the inner angle between v1 and v2
    inner_angles = np.linspace(0, this_great_arc.inner_angle, num=number_points).reshape(number_points, 1)

    # Calculate the Cartesian locations from the first to second points
    return this_great_arc.v1[np.newaxis, :] * np.cos(inner_angles) + \
           this_great_arc.v3[np.newaxis, :] * np.sin(inner_angles) + \
           center_cartesian


class GreatArcPropertiesCartesian:
    def __init__(self, start_cartesian, end_cartesian, center_cartesian):
        """
        Calculate the properties of a great arc between a start point and an
        end point on a sphere.  See the references below for a description of
        the algorithm.
        Parameters
        ----------
        start_cartesian : `~numpy.ndarray`
            Start point expressed as a Cartesian xyz triple.
        end_cartesian : `~numpy.ndarray`
            End point expressed as a Cartesian xyz triple.
        center_cartesian : `~numpy.ndarray`
            Center of the sphere expressed as a Cartesian xyz triple
        References
        ----------
        [1] https://www.mathworks.com/matlabcentral/newsreader/view_thread/277881
        [2] https://en.wikipedia.org/wiki/Great-circle_distance#Vector_version
        """
        self.start_cartesian = start_cartesian
        self.end_cartesian = end_cartesian
        self.center_cartesian = center_cartesian

        # Vector from center to first point
        self.v1 = self.start_cartesian - self.center_cartesian

        # Distance of the first point from the center
        self.r = np.linalg.norm(self.v1)

        # Vector from center to second point
        self.v2 = self.end_cartesian - self.center_cartesian

        # The v3 vector lies in plane of v1 & v2 and is orthogonal to v1
        self.v3 = np.cross(np.cross(self.v1, self.v2), self.v1)
        self.v3 = self.r * self.v3 / np.linalg.norm(self.v3)

        # Inner angle between v1 and v2 in radians
        self.inner_angle = np.arctan2(np.linalg.norm(np.cross(self.v1, self.v2)),
                                      np.dot(self.v1, self.v2))

        # Distance on the sphere between the start point and the end point.
        self.distance = self.r * self.inner_angle


class HelperGreatArcConvertToCartesian:
    def __init__(self, start, end, center=None):
        """
        A helper class that takes the SunPy co-ordinates required to compute a
        great arc and returns Cartesian triples, co-ordinate frame and unit
        information required by other functions.
        Parameters
        ----------
        start : `~astropy.coordinates.SkyCoord`
            Start point.
        end : `~astropy.coordinates.SkyCoord`
            End point.
        center : `~astropy.coordinates.SkyCoord`
            Center of the sphere.
        """
        self.start = start
        self.end = end
        self.center = center

        # Units of the start point
        self.start_unit = self.start.transform_to(frames.Heliocentric).cartesian.xyz.unit

        # Co-ordinate frame
        self.start_frame = self.start.frame

        # Observer details
        self.observer = self.start.observer

        if self.center is None:
            self.c = SkyCoord(0*self.start_unit,
                              0*self.start_unit,
                              0*self.start_unit, frame=frames.Heliocentric)

        self.start_cartesian = self.start.transform_to(frames.Heliocentric).cartesian.xyz.to(self.start_unit).value
        self.end_cartesian = self.end.transform_to(frames.Heliocentric).cartesian.xyz.to(self.start_unit).value
        self.center_cartesian = self.c.transform_to(frames.Heliocentric).cartesian.xyz.to(self.start_unit).value


