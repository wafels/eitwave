#
# Utility functions for AWARE
#
from __future__ import absolute_import, division, print_function
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


def wave_progress_map_by_location_and_fits(mc, results, level=0.0, index=0, minimum_value=0):
    """
    Take an input mapcube and return the location of the wavefront as a
    function of time in a single SunPy map.  The locations are color-coded,
    where the color indicates time.  Locations are calculated by first
    thresholding the data and creating a mask.  That mask is placed in the
    results map.  The result is also convolved with a map denoting if a
    particular location at a time was included in the fit.

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
    #
    wave_progress_data = np.zeros_like(mc[index].data)
    timestamps = []
    # Step through in time
    for im in range(0, len(mc)):
        # Where AWARE detected a wavefront at this time
        detection = mc[im].data > level
        wave_progress_data[detection] = im + minimum_value

        # Where AWARE was able to fit a wavefront at this time

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


#
# Calculates the co-ordinates along great arcs between two specified points
# which are assumed to be on disk.
#

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.coordinates import frames

__all__ = ['GreatArc', 'GreatCircle']


class GreatArc(object):
    """
    Calculate the properties of a great arc at user-specified points between a
    start and end point on a sphere.

    Parameters
    ----------
    start : `~astropy.coordinates.SkyCoord`
        Start point.

    end : `~astropy.coordinates.SkyCoord`
        End point.

    center : `~astropy.coordinates.SkyCoord`
        Center of the sphere.

    points : `None`, `int`, `~numpy.ndarray`
        Number of points along the great arc.  If None, the arc is calculated
        at 100 equally spaced points from start to end.  If int, the arc is
        calculated at "points" equally spaced points from start to end.  If a
        numpy.ndarray is passed, it must be one dimensional and have values
        >=0 and <=1.  The values in this array correspond to parameterized
        locations along the great arc from zero, denoting the start of the arc,
        to 1, denoting the end of the arc.  Setting this keyword on initializing
        a GreatArc object sets the locations of the default points along the
        great arc.

    Methods
    -------
    inner_angles : `~astropy.units.rad`
        Radian angles of the points along the great arc from the start to end
        co-ordinate.

    distances : `~astropy.units`
        Distances of the points along the great arc from the start to end
        co-ordinate.  The units are defined as those returned after transforming
        the co-ordinate system of the start co-ordinate into its Cartesian
        equivalent.

    coordinates : `~astropy.coordinates.SkyCoord`
        Co-ordinates along the great arc in the co-ordinate frame of the
        start point.

    References
    ----------
    [1] https://www.mathworks.com/matlabcentral/newsreader/view_thread/277881
    [2] https://en.wikipedia.org/wiki/Great-circle_distance#Vector_version

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from sunpy.coordinates.utils import GreatArc
    >>> import sunpy.map
    >>> from sunpy.data.sample import AIA_171_IMAGE
    >>> m = sunpy.map.Map(AIA_171_IMAGE)
    >>> a = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=m.coordinate_frame)
    >>> b = SkyCoord(-100*u.arcsec, 800*u.arcsec, frame=m.coordinate_frame)
    >>> great_arc = GreatArc(a, b)
    >>> ax = plt.subplot(projection=m)
    >>> m.plot(axes=ax)
    >>> ax.plot_coord(great_arc.coordinates(), color='c')
    >>> plt.show()

    """

    def __init__(self, start, end, center=None, points=None):
        # Start point of the great arc
        self.start = start

        # End point of the great arc
        self.end = end

        # Parameterized location of points between the start and the end of the
        # great arc.
        # Default parameterized points location.
        self.default_points = np.linspace(0, 1, 100)

        # If the user requests a different set of default parameterized points
        # on initiation of the object, then these become the default.  This
        # allows the user to use the methods without having to specify their
        # choice of points over and over again, while also allowing the
        # flexibility in the methods to calculate other values.
        self.default_points = self._points_handler(points)

        # Units of the start point
        self.distance_unit = self.start.transform_to(frames.Heliocentric).cartesian.xyz.unit

        # Co-ordinate frame
        self.start_frame = self.start.frame

        # Observer
        self.observer = self.start.observer

        # Set the center of the sphere
        if center is None:
            self.center = SkyCoord(0 * self.distance_unit,
                                   0 * self.distance_unit,
                                   0 * self.distance_unit, frame=frames.Heliocentric)

        # Convert the start, end and center points to their Cartesian values
        self.start_cartesian = self.start.transform_to(frames.Heliocentric).cartesian.xyz.to(self.distance_unit).value
        self.end_cartesian = self.end.transform_to(frames.Heliocentric).cartesian.xyz.to(self.distance_unit).value
        self.center_cartesian = self.center.transform_to(frames.Heliocentric).cartesian.xyz.to(self.distance_unit).value

        # Great arc properties calculation
        # Vector from center to first point
        self.v1 = self.start_cartesian - self.center_cartesian

        # Distance of the first point from the center
        self._r = np.linalg.norm(self.v1)

        # Vector from center to second point
        self.v2 = self.end_cartesian - self.center_cartesian

        # The v3 vector lies in plane of v1 & v2 and is orthogonal to v1
        self.v3 = np.cross(np.cross(self.v1, self.v2), self.v1)
        self.v3 = self._r * self.v3 / np.linalg.norm(self.v3)

        # Inner angle between v1 and v2 in radians
        self.inner_angle = np.arctan2(np.linalg.norm(np.cross(self.v1, self.v2)),
                                        np.dot(self.v1, self.v2)) * u.rad

        # Radius of the sphere
        self.radius = self._r * self.distance_unit

        # Distance on the sphere between the start point and the end point.
        self.distance = self.radius * self.inner_angle.value

    def _points_handler(self, points):
        """
        Interprets the points keyword.
        """
        if points is None:
            return self.default_points
        elif isinstance(points, int):
            return np.linspace(0, 1, points)
        elif isinstance(points, np.ndarray):
            if points.ndim > 1:
                raise ValueError('One dimensional numpy ndarrays only.')
            if np.any(points < 0) or np.any(points > 1):
                raise ValueError('All value in points array must be strictly >=0 and <=1.')
            return points
        else:
            raise ValueError('Incorrectly specified "points" keyword value.')

    def inner_angles(self, points=None):
        """
        Calculates the inner angles for the parameterized points along the arc
        and returns the value in radians, from the start co-ordinate to the
        end.

        Parameters
        ----------
        points : `None`, `int`, `~numpy.ndarray`
            If None, use the default locations of parameterized points along the
            arc.  If int, the arc is calculated at "points" equally spaced
            points from start to end.  If a numpy.ndarray is passed, it must be
            one dimensional and have values >=0 and <=1.  The values in this
            array correspond to parameterized locations along the great arc from
            zero, denoting the start of the arc, to 1, denoting the end of the
            arc.

        Returns
        -------
        inner_angles : `~astropy.units.rad`
            Radian angles of the points along the great arc from the start to
            end co-ordinate.

        """
        these_points = self._points_handler(points)
        return these_points.reshape(len(these_points), 1)*self.inner_angle

    def distances(self, points=None):
        """
        Calculates the distance from the start co-ordinate to the end
        co-ordinate on the sphere for all the parameterized points.

        Parameters
        ----------
        points : `None`, `int`, `~numpy.ndarray`
            If None, use the default locations of parameterized points along the
            arc.  If int, the arc is calculated at "points" equally spaced
            points from start to end.  If a numpy.ndarray is passed, it must be
            one dimensional and have values >=0 and <=1.  The values in this
            array correspond to parameterized locations along the great arc from
            zero, denoting the start of the arc, to 1, denoting the end of the
            arc.

        Returns
        -------
        distances : `~astropy.units`
            Distances of the points along the great arc from the start to end
            co-ordinate.  The units are defined as those returned after
            transforming the co-ordinate system of the start co-ordinate into
            its Cartesian equivalent.
        """
        return self.radius * self.inner_angles(points=points).value

    def coordinates(self, points=None):
        """
        Calculates the co-ordinates on the sphere from the start to the end
        co-ordinate for all the parameterized points.  Co-ordinates are
        returned in the frame of the start coordinate.

        Parameters
        ----------
        points : `None`, `int`, `~numpy.ndarray`
            If None, use the default locations of parameterized points along the
            arc.  If int, the arc is calculated at "points" equally spaced
            points from start to end.  If a numpy.ndarray is passed, it must be
            one dimensional and have values >=0 and <=1.  The values in this
            array correspond to parameterized locations along the great arc from
            zero, denoting the start of the arc, to 1, denoting the end of the
            arc.

        Returns
        -------
        arc : `~astropy.coordinates.SkyCoord`
            Co-ordinates along the great arc in the co-ordinate frame of the
            start point.

        """
        return self.cartesian_coordinates(points=points).transform_to(self.start_frame)

    def cartesian_coordinates(self, points=None):
        """
        Calculates the co-ordinates on the sphere from the start to the end
        co-ordinate for all the parameterized points.  Co-ordinates are
        returned in the Cartesian co-ordinate frame.

        Parameters
        ----------
        points : `None`, `int`, `~numpy.ndarray`
            If None, use the default locations of parameterized points along the
            arc.  If int, the arc is calculated at "points" equally spaced
            points from start to end.  If a numpy.ndarray is passed, it must be
            one dimensional and have values >=0 and <=1.  The values in this
            array correspond to parameterized locations along the great arc from
            zero, denoting the start of the arc, to 1, denoting the end of the
            arc.

        Returns
        -------
        arc : `~astropy.coordinates.SkyCoord`
            Co-ordinates along the great arc in the Cartesian co-ordinate frame.

        """
        # Calculate the inner angles
        these_inner_angles = self.inner_angles(points=points)

        # Calculate the Cartesian locations from the first to second points
        great_arc_points_cartesian = (self.v1[np.newaxis, :] * np.cos(these_inner_angles) +
                                      self.v3[np.newaxis, :] * np.sin(these_inner_angles) +
                                      self.center_cartesian) * self.distance_unit

        # Return the coordinates of the great arc between the start and end
        # points
        return SkyCoord(great_arc_points_cartesian[:, 0],
                        great_arc_points_cartesian[:, 1],
                        great_arc_points_cartesian[:, 2],
                        frame=frames.Heliocentric,
                        observer=self.observer)


class GreatCircle(GreatArc):
    def __init__(self, start, end, center=None, points=None):
        """
        Calculate the properties of a great circle at user-specified points
        The great circle passes through two points on a sphere specified by
        the user.  The points returned are in the direction from the start point
        through the end point.

        Parameters
        ----------
        start : `~astropy.coordinates.SkyCoord`
            Start point.

        end : `~astropy.coordinates.SkyCoord`
            End point.

        center : `~astropy.coordinates.SkyCoord`
            Center of the sphere.

        points : `None`, `int`, `~numpy.ndarray`
            Number of points along the great arc.  If None, the arc is calculated
            at 100 equally spaced points from start to end.  If int, the arc is
            calculated at "points" equally spaced points from start to end.  If a
            numpy.ndarray is passed, it must be one dimensional and have values
            >=0 and <=1.  The values in this array correspond to parameterized
            locations along the great arc from zero, denoting the start of the arc,
            to 1, denoting the end of the arc.  Setting this keyword on initializing
            a GreatArc object sets the locations of the default points along the
            great arc.

        Methods
        -------
        inner_angles : `~astropy.units.rad`
            Radian angles of the points along the great arc from the start to end
            co-ordinate.

        distances : `~astropy.units`
            Distances of the points along the great arc from the start to end
            co-ordinate.  The units are defined as those returned after transforming
            the co-ordinate system of the start co-ordinate into its Cartesian
            equivalent.

        coordinates : `~astropy.coordinates.SkyCoord`
            Co-ordinates along the great arc in the co-ordinate frame of the
            start point.

        References
        ----------
        [1] https://www.mathworks.com/matlabcentral/newsreader/view_thread/277881
        [2] https://en.wikipedia.org/wiki/Great-circle_distance#Vector_version

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> from astropy.coordinates import SkyCoord
        >>> import astropy.units as u
        >>> from sunpy.coordinates.utils import GreatCircle
        >>> import sunpy.map
        >>> from sunpy.data.sample import AIA_171_IMAGE
        >>> m = sunpy.map.Map(AIA_171_IMAGE)
        >>> a = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=m.coordinate_frame)
        >>> b = SkyCoord(-100*u.arcsec, 800*u.arcsec, frame=m.coordinate_frame)
        >>> great_circle = GreatCircle(a, b)
        >>> coordinates = great_circle.coordinates(points=1000)
        >>> front_arc_coordinates = coordinates[great_circle.front_arc_indices]
        >>> back_arc_coordinates = coordinates[great_circle.back_arc_indices]
        >>> arc_from_start_to_back = coordinates[0:great_circle.from_front_to_back_index]
        >>> arc_from_back_to_start = coordinates[great_circle.from_back_to_front_index: len(coordinates)-1]
        >>> ax = plt.subplot(projection=m)
        >>> m.plot(axes=ax)
        >>> ax.plot_coord(front_arc_coordinates, color='k', linewidth=5)
        >>> ax.plot_coord(back_arc_coordinates, color='k', linestyle=":")
        >>> ax.plot_coord(arc_from_start_to_back, color='c')
        >>> ax.plot_coord(arc_from_back_to_start, color='r')
        >>> plt.show()
        """
        GreatArc.__init__(self, start, end, center=center, points=points)

        # Set the inner angle to be 2*pi radians, the full circle.
        self.inner_angle = 2 * np.pi * u.rad

        # Boolean array indicating which coordinate is on the front of the disk
        # (True) or on the back (False).
        self.front_or_back = self.cartesian_coordinates().z.value > 0

        # Calculate the indices where the co-ordinates change from being on the
        # front of the disk to the back to the disk.
        self._fob = self.front_or_back.astype(np.int)
        self._change = self._fob[1:] - self._fob[0:-1]
        self.from_front_to_back_index = np.where(self._change == -1)[0][0]
        self.from_back_to_front_index = np.where(self._change == 1)[0][0]

        # Indices of arcs on the front side and the back
        self.front_arc_indices = np.concatenate((np.arange(self.from_back_to_front_index, len(self.coordinates())),
                                                 np.arange(0, self.from_front_to_back_index)))

        self.back_arc_indices = np.arange(self.from_front_to_back_index + 1,
                                          self.from_back_to_front_index)
