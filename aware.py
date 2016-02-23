#
# Demonstration AWARE algorithm
#
import os
from copy import deepcopy
import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
from scipy.misc import bytescale
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk
from skimage.morphology.selem import ellipse
from skimage.filter.rank import median
from sunpy.map import Map
from sunpy.time import parse_time
import aware_utils
import aware_plot
import mapcube_tools
import astropy.units as u

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# The factor below is the circumference of the sun in meters kilometers divided
# by 360 degrees.
solar_circumference_per_degree_in_km = aware_utils.solar_circumference_per_degree.to('km/deg') * u.degree


def dump_images(mc, directory, name):
    for im, m in enumerate(mc):
        fname = '%s_%05d.png' % (name, im)
        dump_image(m.data, directory, fname)


def dump_image(img, directory, name):
    ndir = os.path.expanduser('~/eitwave/img/%s/' % directory)
    if not(os.path.exists(ndir)):
        os.makedirs(ndir)
    plt.ioff()
    plt.imshow(img)
    plt.savefig(os.path.join(ndir, name))


#
# Some potential improvements
#
# 1. Do the median filtering and the closing on multiple length-scales
#    Could add up the results at multiple length-scales to get a better idea of
#    where the wavefront is
#
# 2. Apply the median and morphological operations on the 3 dimensional
#    datacube, to take advantage of previous and future observations.
#
def processing(mc, radii=[[11, 11]*u.degree],
               histogram_clip=[0.0, 99.], func=np.sqrt, develop=False,
               verbose=True):
    """
    Image processing steps used to isolate the EUV wave from the data.  Use
    this part of AWARE to perform the image processing steps that segment
    propagating features that brighten new pixels as they propagate.

    Parameters
    ----------

    mc : sunpy.map.MapCube
    radii : list of lists. Each list contains a pair of numbers that describe the
    radius of the median filter and the closing operation
    histogram_clip
    func
    """

    # Histogram the data
    mc_data = func(mc.as_array())
    clip_limit = np.percentile(mc_data[np.isfinite(mc_data)], histogram_clip)

    # Define the disks that will be used on all the images.
    # The first disk in each pair is the disk that is used by the median
    # filter.  The second disk is used by the morphological closing
    # operation.
    disks = []
    for r in radii:
        e1 = (r[0]/mc[0].scale.x).to('pixel').value  # median ellipse width - across wavefront
        e2 = (r[0]/mc[0].scale.y).to('pixel').value  # median ellipse height - along wavefront

        e3 = (r[1]/mc[0].scale.x).to('pixel').value  # closing ellipse width - across wavefront
        e4 = (r[1]/mc[0].scale.y).to('pixel').value  # closing ellipse height - along wavefront

        disks.append([disk(e1), disk(e3)])

    # For the dump images
    rstring = ''
    for r in radii:
        z = '%i_%i__' % (r[0].value, r[1].value)
        rstring += z

    # Calculate the persistence
    new = mapcube_tools.persistence(mc)
    if develop:
        dump_images(new, rstring, '%s_1_persistence' % rstring)

    # Calculate the running difference
    new = mapcube_tools.running_difference(new)
    if develop:
        dump_images(new, rstring, '%s_2_rdiff' % rstring)

    # Storage for the processed mapcube.
    new_mc = []

    # Get each map out of the cube an clean it up to better isolate the wave
    # front
    for im, m in enumerate(new):
        if verbose:
            print("  AWARE: processing map %i out of %i" % (im, len(new)))
        # Dump images - identities
        ident = (rstring, im)

        # Rescale the data using the input function, and subtract the lower
        # clip limit so it begins at zero.
        f_data = func(m.data) - clip_limit[0]

        # Replace the nans with zeros - the reason for doing this rather than
        # something more sophisticated is that nans will not contribute
        # greatly to the final answer.  The nans are put back in at the end
        # and get carried through in the maps.
        nans_here = np.logical_not(np.isfinite(f_data))
        nans_replaced = deepcopy(f_data)
        nans_replaced[nans_here] = 0.0

        # Byte scale the data - recommended input type for the median.
        new_data = bytescale(nans_replaced)
        if develop:
            dump_image(new_data, rstring, '%s_345_bytscale_%i_%05d.png' % (rstring, im, im))

        # Final image used to measure the location of the wave front
        final_image = np.zeros_like(new_data, dtype=np.float32)

        # Clean the data to isolate the wave front.
        for j, d in enumerate(disks):
            # Get rid of noise by applying the median filter.  Although the
            # input is a byte array make sure that the output is a floating
            # point array for use with the morphological closing operation.
            new_d = 1.0*median(new_data, d[0])
            if develop:
                dump_image(new_d, rstring, '%s_6_median_%i_%05d.png' % (rstring, radii[j][0].value, im))

            # Apply the morphological closing operation to rejoin separated
            # parts of the wave front.
            new_d = closing(new_d, d[1])
            if develop:
                dump_image(new_d, rstring, '%s_7_closing_%i_%05d.png' % (rstring, radii[j][1].value, im))

            # Further insurance that we get floating point arrays which are
            # summed below.
            final_image += new_d*1.0

        if develop:
            dump_image(final_image, rstring, '%s_final_%05d.png' % ident)

        # New mapcube list
        new_mc.append(Map(ma.masked_array(final_image, mask=nans_here), m.meta))

    # Return the cleaned mapcube
    return Map(new_mc, cube=True)


def _my_odr_quadratic_function(B, x):
    return B[0] * x ** 2 + B[1] * x + B[2]


def _get_times_from_start(mc, start_date=None):
    # Get the times of the images
    if start_date is None:
        start_time = parse_time(mc[0].date)
    else:
        start_time = parse_time(start_date)
    return np.asarray([(parse_time(m.date) - start_time).seconds for m in mc])


"""
@mapcube_tools.mapcube_input
def get_arcs(unraveled_data, example_map, originating_event_time=None, verbose=False):
    Get all the arcs from an unraveled mapcube.

    :param unraveled:
    :param originating_event_time:
    :param verbose:
    :return:


    nlon = np.int(example_map[0].dimensions[0].value)
    results = []
    for lon in range(0, nlon):
        if verbose:
            print('Analyzing %i out of %i arcs.' % (lon, nlon))
        results.append(get_arc(unraveled_data[:, lon, :], originating_event_time=originating_event_time))
    return results
"""


def arc_as_fit(arc, error_choice='std', position_choice='average'):
    if position_choice == 'average':
        position = arc.average_position()
    if position_choice == 'maximum':
        position = arc.maximum_position()
    if error_choice == 'std':
        error = arc.wavefront_position_error_estimate_standard_deviation()
    if error_choice == 'width':
        error = arc.wavefront_position_error_estimate_width(position_choice)
    return arc.times, position, error


def arcs_as_fit(arcs, error_choice='std', position_choice='average'):
    results = []
    for arc in arcs:
        results.append(arc_as_fit(arc,
                                  error_choice=error_choice,
                                  position_choice=position_choice))
    return results


def dynamic(times, position, position_error, n_degree=1, ransac_kwargs=None):
    """
    :param times:
    :param position:
    :param position_error:
    :return:
    """
    return FitPosition(times, position, position_error, n_degree=n_degree,
                       ransac_kwargs=ransac_kwargs)


def dynamics(arcs_as_fit, ransac_kwargs=None, n_degree=1):
    """
    Measurement of the progress of the wave across the disk.  This part of
    AWARE generates information concerning the dynamics of the wavefront.
    """
    results = []
    for i, arc_as_fit in enumerate(arcs_as_fit):
        results.append(dynamic(arc_as_fit[0], arc_as_fit[1], arc_as_fit[2],
                               n_degree=n_degree,
                               ransac_kwargs=ransac_kwargs))
    return results


def average_position(data, times, latitude):
    """
    Calculate the average position of the wavefront
    :param data:
    :param times:
    :param latitude:
    :return:
    """
    nt = len(times)

    # Average position
    pos = np.zeros(nt)
    for i in range(0, nt):
        emission = data[::-1, i]
        summed_emission = np.nansum(emission)
        pos[i] = np.nansum(emission * latitude) / summed_emission
    return pos


def maximum_position(data, times, latitude):
    """
    Calculate the maximum position of the wavefront
    :param data:
    :param times:
    :param latitude:
    :return:
    """
    nt = len(times)

    # Maximum Position
    pos = np.zeros(nt)
    for i in range(0, nt):
        emission = data[::-1, i]
        pos[i] = latitude[np.nanargmax(emission)]
    return pos


def wavefront_position_error_estimate_standard_deviation(data, times, latitude):
    """
    Calculate the standard deviation of the width of the wavefornt
    :param data:
    :param times:
    :param latitude:
    :return:
    """
    nt = len(times)
    error = np.zeros(nt)
    for i in range(0, nt):
        emission = data[::-1, i]
        summed_emission = np.nansum(emission)
        error[i] = np.nanstd(emission * latitude) / summed_emission
    return error


def wavefront_position_error_estimate_width(data, times, lat_bin, position_choice):
    """
    Calculate the standard deviation of the width of the wavefornt
    :param data:
    :param times:
    :param latitude:
    :return:
    """
    single_pixel_std = lat_bin * np.sqrt(1.0 / 12.0)
    nt = len(times)
    error = np.zeros(nt)
    for i in range(0, nt):
        emission = data[::-1, i]
        nonzero_emission = np.nonzero(emission)
        # Maximum width of the wavefront
        if len(nonzero_emission[0]) > 0:
            max_width = lat_bin * (1 + nonzero_emission[0][-1] - nonzero_emission[0][0])
        else:
            max_width = 0.0

        if position_choice == 'maximum':
            # In this case the location of the wave is determined by
            # looking at the pixel with the maximum of the emission
            # and at the extent of the wave.  This means that there
            # are three pixel widths to consider.  The sources of
            # error are summed as the square root of the sum of
            # squares.
            single_pixel_factor = 3.0
        else:
            # In this case the location of the wave is determined by
            # determining the width of wavefront in pixels.  This
            # means that there are two pixel widths to consider.
            # The sources of error are summed as the square root of
            # the sum of squares.
            single_pixel_factor = 2.0
        error[i] = np.sqrt(max_width ** 2 + single_pixel_factor * single_pixel_std ** 2)

    return error


class Arc:
    """
    Object to store data on the emission along each arc as a function of time
    data : ndarray of size (nlat, nt)
    times : ndarray of time in seconds from zero
    latitude : ndarray of the latitude bins of the unraveled array
    """
    def __init__(self, data, times, latitude, offset, title):

        self.data = data
        self.times = times
        self.latitude = latitude
        self.offset = offset
        self.lat_bin = self.latitude[1] - self.latitude[0]
        self.title = title

    def average_position(self):
        return average_position(self.data, self.times, self.latitude)

    def maximum_position(self):
        return maximum_position(self.data, self.times, self.latitude)

    def wavefront_position_error_estimate_standard_deviation(self):
        return wavefront_position_error_estimate_standard_deviation(self.data, self.times, self.latitude)

    def wavefront_position_error_estimate_width(self, position_choice):
        return wavefront_position_error_estimate_width(self.data, self.times, self.lat_bin, position_choice)

    def peek(self):
        return aware_plot.arc(self)


class FitPosition:
    """
    An object that performs a fit to an AWARE Arc object, and holds the full
    details on how the fit was performed and its results.

    Parameters
    ----------
    times : one-dimensional ndarray of size nt

    position : one-dimensional ndarray of size nt

    error : one-dimensional ndarray of size nt

    n_degree : int
        degree of the polynomial to fit

    ransac_kwargs : dict
        keywords for the RANSAC algorithm

    """

    def __init__(self, times, position, error, n_degree=2, verbose=False,
                 ransac_kwargs=None):
        self.times = times
        self.nt = len(times)
        self.position = position
        self.error = error
        self.ransac_kwargs = ransac_kwargs
        self.n_degree = n_degree

        # Is the arc fit-able?  Assume that it is.
        self.fit_able = True

        # Has the arc been fitted?
        self.fitted = False

        # Find if we have enough points to do a quadratic fit
        # Simple test to see how much the first few points affect the fit
        self.position_is_finite = np.isfinite(self.position)
        self.at_least_one_nonzero_location = np.any(np.abs(self.position[self.position_is_finite]) > 0.0)
        self.error_is_finite = np.isfinite(self.error)
        self.error_is_above_zero = self.error > 0
        self.defined = self.position_is_finite * \
                       self.at_least_one_nonzero_location * \
                       self.error_is_finite * \
                       self.error_is_above_zero

        if self.ransac_kwargs is not None:
            # Find inliers using RANSAC, if there are enough points
            if np.sum(self.defined) > 3:
                this_x = deepcopy(self.times[self.defined])
                this_y = deepcopy(self.position[self.defined])
                median_error = np.median(self.error[self.defined])
                model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(residual_threshold=median_error))
                try:
                    model.fit(this_x.reshape((len(this_x), 1)), this_y)
                    self.inlier_mask = np.asarray(model.named_steps['ransacregressor'].inlier_mask_)
                    self.ransac_success = True
                except ValueError:
                    self.ransac_success = False
                    self.inlier_mask = np.ones(np.sum(self.defined), dtype=bool)
        else:
            self.ransac_success = None
            self.inlier_mask = np.ones(np.sum(self.defined), dtype=bool)

        # Are there enough points to do a fit?
        if np.sum(self.inlier_mask) <= 3:
            self.fit_able = False

        # Perform a fit if there enough points
        if self.fit_able:
            # Get the locations where the location is defined
            self.locf = self.position[self.defined][self.inlier_mask]
            # Get the standard deviation where the location is defined
            self.errorf = self.error[self.defined][self.inlier_mask]
            # Get the times where the location is defined
            self.timef = self.times[self.defined][self.inlier_mask]

            # Do the quadratic fit to the data
            try:
                self.poly_fit, self.covariance = np.polyfit(self.timef, self.locf, self.n_degree, w=1.0/(self.errorf ** 2), cov=True)
                self.fitted = True

                # Calculate the best fit line assuming no error in the input
                # times
                self.best_fit = np.polyval(self.poly_fit, self.timef)

                # Convert to km/s
                self.vel_index = self.n_degree - 1
                self.velocity = self.poly_fit[self.vel_index] * solar_circumference_per_degree_in_km / u.s
                self.velocity_error = np.sqrt(self.covariance[self.vel_index, self.vel_index]) * solar_circumference_per_degree_in_km / u.s

                # Convert to km/s/s
                if self.n_degree >= 2:
                    self.acc_index = self.n_degree - 2
                    self.acceleration = 2 * self.poly_fit[self.acc_index] * solar_circumference_per_degree_in_km / u.s / u.s
                    self.acceleration_error = 2 * np.sqrt(self.covariance[self.acc_index, self.acc_index]) * solar_circumference_per_degree_in_km / u.s / u.s
                else:
                    self.acceleration = None
                    self.acceleration_error = None

                # Reduced chi-squared.
                self.rchi2 = (1.0 / (1.0 * (len(self.timef) - (1.0 + self.n_degree)))) * np.sum(((self.best_fit - self.locf) / self.errorf) ** 2)

                # Log likelihood
                self.log_likelihood = np.sum(-0.5*np.log(2*np.pi*self.errorf**2) - 0.5*((self.best_fit - self.locf) / self.errorf) ** 2)

                # AIC
                self.AIC = 2 * self.n_degree - 2 * self.log_likelihood

                # BIC
                self.BIC = -2 * self.log_likelihood + self.n_degree * np.log(self.timef.size)


                """
                # Calculate the Long et al (2014) score
                self.long_score = aware_utils.score_long(self.nlat,
                                                         self.defined[self.inlier_mask],
                                                         self.velocity,
                                                         self.acceleration,
                                                         self.errorf,
                                                         self.locf,
                                                         self.nt)

                # Fractional duration of the arc
                self.arc_duration_fraction = aware_utils.arc_duration_fraction(self.defined, self.nt)
                """

            except (LA.LinAlgError, ValueError):
                # Error in the fitting algorithm
                self.fitted = False

            """
            # Successful OLS fit - try the ODR fit
            if self.fitted:


                # Calculate the best fit line assuming that the input time is
                # a mean time that represents the spread of time used to create
                # the input images
                myodr = odr.ODR(self.odr_data, _my_odr_quadratic_function, beta0=self.poly_fit)
                self.my_odr_output = myodr.run()
            """
