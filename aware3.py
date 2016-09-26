#
# Demonstration AWARE algorithm
#
# This file contains the image processing, arc definition and model fitting
# routines that comprise the AWARE algorithm.
#
#
from copy import deepcopy
import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
from scipy.ndimage.filters import median_filter
from scipy.ndimage import grey_closing
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from skimage.morphology import disk
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

import astropy.units as u

from sunpy.map import Map
from sunpy.time import parse_time

import aware_utils
import aware_constants
import mapcube_tools


# The factor below is the circumference of the sun in meters kilometers divided
# by 360 degrees.
solar_circumference_per_degree_in_km = aware_constants.solar_circumference_per_degree.to('km/deg') * u.degree


#
# AWARE:  image processing
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
@mapcube_tools.mapcube_input
def processing(mc, radii=[[11, 11]*u.degree],
               clip_limit=None,
               histogram_clip=[0.0, 99.],
               func=np.sqrt,
               develop=None):
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
    clip_limit :
    func :
    develop :

    """

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
    if develop is not None:
        filename = develop + '_persistence'
        print('Writing movie to {:s}.'.format(filename))
        aware_utils.write_movie(new, filename)

    # Calculate the running difference
    new = mapcube_tools.running_difference(new)
    if develop is not None:
        filename = develop + '_running_difference'
        print('Writing movie to {:s}.'.format(filename))
        aware_utils.write_movie(new, filename)

    # Storage for the processed mapcube.
    new_mc = []

    # Only want positive differences, so everything lower than zero
    # should be set to zero
    mc_data = func(new.as_array())
    mc_data[mc_data < 0.0] = 0.0

    # Clip the data to be within a range, and then normalize it.
    if clip_limit is None:
        cl = np.nanpercentile(mc_data, histogram_clip)
    mc_data[mc_data > cl[1]] = cl[1]
    mc_data = (mc_data - cl[0]) / (cl[1]-cl[0])

    # Get rid of NaNs
    nans_here = np.logical_not(np.isfinite(mc_data))
    nans_replaced = deepcopy(mc_data)
    nans_replaced[nans_here] = 0.0

    # Clean the data to isolate the wave front.  Use three dimensional
    # operations from scipy.ndimage.  This approach should get rid of
    # more noise and have better continuity in the time-direction.
    final = np.zeros_like(mc_data, dtype=np.float32)

    # Do the cleaning and isolation operations on multiple length-scales,
    # and add up the final results.
    for j, d in enumerate(disks):
        pancake = np.swapaxes(np.tile(d[0], (3, 1, 1)), 0, -1)
        nr = deepcopy(nans_replaced)

        print('\n', nr.shape, pancake.shape, '\n', 'started median filter' )
        nr = 1.0*median_filter(nr, footprint=pancake)

        print(' started grey closing')
        nr = 1.0*grey_closing(nr, footprint=pancake)

        # Sum all the
        final += nr*1.0

    for i, m in enumerate(new):
        new_mc.append(Map(ma.masked_array(final[:, :, i],
                                          mask=nans_here[:, :, i]),
                          m.meta))

    # Return the cleaned mapcube
    return Map(new_mc, cube=True)


#
###############################################################################
#
# AWARE: defining the arcs
#
def gaussian(x, amplitude, position, width):
    """
    Function used to localize the wavefront
    :param x: independent variable
    :param amplitude: amplitude of the Gaussian
    :param position: position of the Gaussian
    :param width: width of the Gaussian
    :return: the Gaussian function over the range of 'x'
    """
    onent = (x-position)/width
    term1 = amplitude/(width * np.sqrt(2*np.pi))
    return term1 * np.exp(-0.5*onent**2)


@mapcube_tools.mapcube_input
def get_times_from_start(mc, start_date=None):
    # Get the times of the images
    if start_date is None:
        start_time = parse_time(mc[0].date)
    else:
        start_time = parse_time(start_date)
    return np.asarray([(parse_time(m.date) - start_time).seconds for m in mc]) * u.s


@u.quantity_input(times=u.s, latitude=u.degree)
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
        pos[i] = np.nansum(emission * latitude.to(u.degree).value) / summed_emission
    return pos * u.degree


@u.quantity_input(times=u.s, latitude=u.degree)
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
        pos[i] = latitude[np.nanargmax(emission)].to(u.degree).value
    return pos * u.degree


def estimate_fwhm(x, y, maximum, arg_maximum):
    half_max = 0.5*maximum
    above = y > half_max
    x_lhs = np.min(x[above])
    x_rhs = np.max(x[above])
    if x_lhs < arg_maximum < x_rhs:
        return x_rhs - x_lhs
    else:
        return None


@u.quantity_input(times=u.s, latitude=u.degree)
def position_and_error_by_fitting_gaussian(data, times, latitude):
    """
    Calculate the position of the wavefront by fitting a Gaussian profile.
    :param data:
    :param times:
    :param latitude:
    :return:
    """
    nt = len(times)
    position = np.zeros(nt)
    error = np.zeros_like(position)
    latitude_value = latitude.to(u.degree).value
    latitude_where_finite = np.isfinite(latitude_value)
    for i in range(0, nt):
        emission = data[::-1, i]
        fit_here = np.logical_or(latitude_where_finite, np.isfinite(emission))
        try:
            amplitude_estimate = np.max(emission[fit_here])
            position_estimate = latitude[np.argmax(emission[fit_here])]
            width_estimate = estimate_fwhm(latitude[fit_here], emission[fit_here], amplitude_estimate, position_estimate)
            p0 = [amplitude_estimate, position_estimate, width_estimate]
            popt, pcov = curve_fit(gaussian, latitude_value[fit_here], emission[fit_here], p0=p0)
            position[i] = popt[1]
            error[i] = pcov[1]
        except RuntimeError:
            position[i] = None
            error[i] = None
        finally:
            position[i] = None
            error[i] = None

    return position*u.degree, error*u.degree


@u.quantity_input(times=u.s, latitude=u.degree)
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
        try:
            error[i] = np.nanstd(emission * latitude.to(u.degree).value) / summed_emission
        except TypeError:
            error[i] = np.nan
    return error * u.degree


@u.quantity_input(times=u.s, lat_bin=u.degree/u.pix)
def wavefront_position_error_estimate_width(data, times, lat_bin, position_choice='maximum'):
    """
    Calculate the standard deviation of the width of the wavefornt
    :param data:
    :param times:
    :param latitude:
    :return:
    """
    single_pixel_std = np.sqrt(1.0 / 12.0)
    nt = len(times)
    error = np.zeros(nt)
    for i in range(0, nt):
        emission = data[::-1, i]
        nonzero_emission = np.nonzero(emission)
        # Maximum width of the wavefront
        if len(nonzero_emission[0]) > 0:
            max_width = 1 + nonzero_emission[0][-1] - nonzero_emission[0][0]
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

    return error * u.pix * lat_bin


class Arc:
    """
    Object to store data on the emission along each arc as a function of time
    data : ndarray of size (nlat, nt)
    times : ndarray of time in seconds from zero
    latitude : ndarray of the latitude bins of the unraveled array
    """
    @u.quantity_input(times=u.s, latitude=u.degree, longitude=u.degree)
    def __init__(self, data, times, latitude, longitude, title=None):

        self.data = data
        self.times = times
        self.latitude = latitude
        self.lat_bin = (self.latitude[1] - self.latitude[0])/u.pix
        self.longitude = longitude
        if title is None:
            self.title = 'longitude=%s' % str(self.longitude)
        else:
            self.title = title

    def average_position(self):
        return average_position(self.data, self.times, self.latitude)

    def maximum_position(self):
        return maximum_position(self.data, self.times, self.latitude)

    def position_and_error_by_fitting_gaussian(self):
        return position_and_error_by_fitting_gaussian(self.data, self.times, self.latitude)

    def wavefront_position_error_estimate_standard_deviation(self):
        return wavefront_position_error_estimate_standard_deviation(self.data, self.times, self.latitude)

    def wavefront_position_error_estimate_width(self, position_choice='maximum'):
        return wavefront_position_error_estimate_width(self.data, self.times, self.lat_bin, position_choice=position_choice)

    def peek(self):
        plt.imshow(self.data, aspect='auto', interpolation='none',
                   extent=[self.times[0].to(u.s).value,
                           self.times[-1].to(u.s).value,
                           self.latitude[0].to(u.degree).value,
                           self.latitude[-1].to(u.degree).value])
        plt.xlim(0, self.times[-1].to(u.s).value)
        if self.times[0].to(u.s).value > 0.0:
            plt.fill_betweenx([self.latitude[0].to(u.degree).value,
                               self.latitude[-1].to(u.degree).value],
                              self.times[0].to(u.s).value,
                              hatch='X', facecolor='w', label='not observed')
        plt.ylabel('degrees of arc from first measurement')
        plt.xlabel('time since originating event (seconds)')
        plt.title('arc: ' + self.title)
        plt.legend(framealpha=0.5)
        plt.show()
        return None


#
###############################################################################
#
# AWARE: arcs to arrays used for fitting
#
class ArcSummary:

    def __init__(self, arc, error_choice='std', position_choice='average'):
        """
        Take an Arc object and calculate the time of the observation,
        the position of the wavefront and the error in that position.

        Parameters
        ----------
        :param arc: an AWARE arc object
            A two-dimensional array that shows the evolution of the intensity of
            the wave as a function of time and latitude along the arc.
        :param error_choice:
            how to measure the error in the position
        :param position_choice:
            how to measure the position
        :return: list
            the time of each wavefront measurement, the position of the wavefront
            and the error in the position.
        """
        self.position_choice = position_choice
        self.error_choice = error_choice
        self.title = arc.title
        self.longitude = arc.longitude

        if self.error_choice == 'std':
            self.position_error = arc.wavefront_position_error_estimate_standard_deviation()
        elif self.error_choice == 'width':
            self.position_error = arc.wavefront_position_error_estimate_width(self.position_choice)
        else:
            raise ValueError('Unrecognized error choice.')

        self.times = arc.times
        if self.position_choice == 'average':
            self.position = arc.average_position()
        elif self.position_choice == 'maximum':
            self.position = arc.maximum_position()
        elif self.position_choice == 'Gaussian':
            self.position, self.position_error = arc.position_by_fitting_gaussian()
        else:
            raise ValueError('Unrecognized position choice.')

    def peek(self):
        plt.errorbar(self.times.to(u.s).value,
                     self.position.to(u.degree).value,
                     yerr=self.position_error.to(u.degree).value,
                     label="{:s}-{:s}".format(self.position_choice, self.error_choice))
        plt.xlabel('times (s)')
        plt.ylabel('position (degrees')
        plt.legend(framealpha=0.5)
        plt.title(self.title)


#
###############################################################################
#
# AWARE: fitting models to the arcs
#
@u.quantity_input(times=u.s, position=u.degree, position_error=u.degree)
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


#
# Log likelihood function.  In this case we want the product of normal
# distributions.
#
def lnlike(variables, x, data, model_function, sigma):
    """
    Log likelihood of the data given a model.

    :param variables: array like, variables used by model_function
    :param x: the independent variable
    :param data: the dependent variable - this is the data we are trying to fit
    with the model
    :param model_function: the model that we are using to fit the power
    spectrum
    :return: the log likelihood of the data given the model.
    """
    model = model_function(variables, x)
    return -np.sum(np.log(np.sqrt(2*np.pi*sigma**2))) - np.sum(((data-model)**2)/(2*sigma**2))


#
# Fit the input model to the data.
#
def minimization(x, data, errors, model_function, initial_guess, bounds, method):
    """
    A wrapper around scipy's minimization function setting up arbitrary
    function fits.

    x : array-like
    The independent variable.

    data : ndarray
    The data we are trying to fit with the model i.e., we are assuming that
    data ~ model(x, model_variables).

    model_function : Python function
    The model that we are fitting to the data.

    initial_guess : array-like
    An initial guess to the model parameters, in the same order as 'x'.

    errors : array-like
    Errors in the measurement of the data, same length as data

    method : the function minimization method used.
    The method used to

    Returns
    -------
    The output from the minimization routine.
    """
    nll = lambda *args: -lnlike(*args)
    args = (x, data, model_function, errors)
    return minimize(nll, initial_guess, args=args, bounds=bounds, method=method)


class FitPosition:
    """
    An object that performs a fit to the position of an EUV wave (along a
    single arc) over the duration of the observation.  This object holds
    the full details of what was fit and how.

    The algorithm determines if there is sufficient information in the
    parameters to perform a fit.  If so, then a fit is attempted.

    Parameters
    ----------
    times : one-dimensional Quantity array of size nt with units convertible
            to seconds

    position : one-dimensional Quantity array of size nt with units convertible
            to degrees of arc

    error : one-dimensional Quantity array of size nt with units convertible
            to degrees of arc

    n_degree : int
        degree of the polynomial to fit

    ransac_kwargs : dict
        keywords for the RANSAC algorithm

    arc_identity : any object
        a free-form holder for identity information for this arc.

    error_tolerance_kwargs : dict
        keywords controlling which positions have a tolerable about of error.
        Only positions that satisfy these conditions go on to be fit.

    """

    @u.quantity_input(times=u.s, position=u.degree, error=u.degree)
    def __init__(self, times, position, error, n_degree=2, ransac_kwargs=None,
                 error_tolerance_kwargs=None, arc_identity=None,
                 fit_method='poly_fit', constrained_fit_method='L-BFGS-B',
                 cvt_factor=2.0):

        self.times = times.to(u.s).value
        self.nt = len(times)
        self.position = position.to(u.degree).value
        self.error = error.to(u.degree).value
        self.n_degree = n_degree
        self.ransac_kwargs = ransac_kwargs
        self.error_tolerance_kwargs = error_tolerance_kwargs
        self.arc_identity = arc_identity
        self.fit_method = fit_method
        self.constrained_fit_method = constrained_fit_method
        self.cvt_factor = cvt_factor

        # At the outset, assume that the arc is able to be fit.
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
            # Find inliers using RANSAC, if there are enough points.  RANSAC is
            # used to help find a large set of points that lie close to the
            # requested polynomial
            if np.sum(self.defined) > 3:
                this_x = deepcopy(self.times[self.defined])
                this_y = deepcopy(self.position[self.defined])
                self.ransac_residual_error = np.median(self.error[self.defined])
                model = make_pipeline(PolynomialFeatures(self.n_degree), RANSACRegressor(residual_threshold=self.ransac_residual_error))
                try:
                    model.fit(this_x.reshape((len(this_x), 1)), this_y)
                    self.inlier_mask = np.asarray(model.named_steps['ransacregressor'].inlier_mask_)
                    self.ransac_success = True
                except ValueError:
                    self.inlier_mask = np.ones(np.sum(self.defined), dtype=bool)
                    self.ransac_success = False
            else:
                self.ransac_success = None
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

            # Get the error where the location is defined
            self.errorf = self.error[self.defined][self.inlier_mask]

            # Errors which are too small can really bias the fit.  The entire
            # fit can be pulled to take into account a couple of really bad
            # points.  This section attempts to fix that by giving those points
            # a user-defined value.
            if 'threshold_error' in self.error_tolerance_kwargs.keys():
                self.threshold_error = self.error_tolerance_kwargs['threshold_error'](self.error[self.defined])
                if 'function_error' in self.error_tolerance_kwargs.keys():
                    self.errorf[self.errorf < self.threshold_error] = self.error_tolerance_kwargs['function_error'](self.error[self.defined])
                else:
                    self.errorf[self.errorf < self.threshold_error] = self.threshold_error

            # Get the times where the location is defined
            self.timef = self.times[self.defined][self.inlier_mask]

            # Do the fit to the data
            try:
                # Where the velocity will be stored in the final results
                self.vel_index = self.n_degree - 1

                #
                # Polynomial and conditional fits to the data
                #
                if self.fit_method == 'poly_fit' or self.fit_method == 'conditional':
                    self.estimate, self.covariance = np.polyfit(self.timef, self.locf, self.n_degree, w=1.0/(self.errorf ** 2), cov=True)

                    # If the code gets this far, then we can assume that a fit
                    # has completed.
                    self.fitted = True
                    self.best_fit = np.polyval(self.estimate, self.timef)
                    ve = np.abs(np.sqrt(self.covariance[self.vel_index, self.vel_index]))
                    self.conditional_velocity_trigger = self.estimate[self.vel_index] + self.cvt_factor * ve
                    if self.fit_method == 'conditional' and self.conditional_velocity_trigger < 0:
                        self.constrained_minimization()
                        self.fit_method = 'conditional (constrained)'
                #
                # Constrained fit to the data
                #
                if self.fit_method == 'constrained':
                    self.constrained_minimization()

                # Convert to km/s
                self.velocity = self.estimate[self.vel_index] * solar_circumference_per_degree_in_km / u.s
                self.velocity_error = np.sqrt(self.covariance[self.vel_index, self.vel_index]) * solar_circumference_per_degree_in_km / u.s

                # Convert to km/s/s
                if self.n_degree >= 2:
                    self.acc_index = self.n_degree - 2
                    self.acceleration = 2 * self.estimate[self.acc_index] * solar_circumference_per_degree_in_km / u.s / u.s
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

                # Calculate the Long et al (2014) score
                self.long_score = aware_utils.ScoreLong(self.velocity,
                                                        self.acceleration,
                                                        self.errorf,
                                                        self.locf,
                                                        self.nt)

                # The fraction of the input arc was actually used in the fit
                self.arc_duration_fraction = len(self.timef) / (1.0 * self.nt)

            except (LA.LinAlgError, ValueError):
                # Error in the fitting algorithm
                self.fitted = False

    def constrained_minimization(self):
        constrained_model = np.polyval
        constrained_initial_guess = np.polyfit(self.timef, self.locf, self.n_degree, w=1.0/(self.errorf ** 2))

        # Generate the bounds - the initial velocity cannot be
        # less than zero.
        if self.n_degree == 1:
            constrained_bounds = ((0.0, None), (None, None))
        if self.n_degree == 2:
            constrained_bounds = ((None, None), (0.0, None), (None, None))

        # Do the minimization with bounds on the velocity.  The
        # initial velocity is not allowed to go below zero, as this
        # would correspond to the wave initially moving backwards.
        constrained_result = minimization(self.timef,
                                          self.locf,
                                          self.errorf,
                                          constrained_model,
                                          constrained_initial_guess,
                                          constrained_bounds,
                                          self.constrained_fit_method)
        self.estimate = constrained_result['x']
        self.covariance = constrained_result['hess_inv'].todense()  # Error estimate?
        self.fitted = constrained_result['success']
        self.best_fit = constrained_model(self.estimate, self.timef)

    def peek(self):
        """
        A summary plot of the results the fit.
        """

        # Calculate positions for plotting text
        ny_pos = 3
        y_pos = np.zeros(ny_pos)
        for i in range(0, ny_pos):
            y_min = np.nanmin(self.position - self.error)
            y_max = np.nanmax(self.position + self.error)
            y_pos[i] = y_min + i * (y_max - y_min) / (1.0 + 1.0*ny_pos)
        x_pos = np.zeros_like(y_pos)
        x_pos[:] = np.min(self.times) + 0.5*(np.max(self.times) - np.min(self.times))

        # Show all the data
        plt.errorbar(self.times, self.position, yerr=self.error,
                     color='k', label='all data')

        # Information labels
        plt.xlabel('times (seconds) [{:n} images]'.format(len(self.times)))
        plt.ylabel('degrees of arc from initial position')
        plt.title(str(self.arc_identity))
        plt.text(x_pos[0], y_pos[0], 'n={:n}'.format(self.n_degree))

        # Show areas where the position is not defined
        at_least_one_not_defined = False
        for i in range(0, self.nt):
            if not self.defined[i]:
                if i == 0:
                    t0 = self.times[0]
                    t1 = 0.5*(self.times[i] + self.times[i+1])
                elif i == self.nt-1:
                    t0 = 0.5*(self.times[i-1] + self.times[i])
                    t1 = self.times[self.nt-1]
                else:
                    t0 = 0.5*(self.times[i-1] + self.times[i])
                    t1 = 0.5*(self.times[i] + self.times[i+1])
                if not at_least_one_not_defined:
                    at_least_one_not_defined = True
                    plt.axvspan(t0, t1, color='b', alpha=0.1, edgecolor='none', label='no detection')
                else:
                    plt.axvspan(t0, t1, color='b', alpha=0.1, edgecolor='none')

        if self.fitted:
            # Show the data used in the fit
            plt.errorbar(self.timef, self.locf, yerr=self.errorf,
                         marker='o', linestyle='None', color='r',
                         label='data used in fit')

            # Show the best fit arc
            plt.plot(self.timef, self.best_fit, color='r', label='best fit ({:s})'.format(self.fit_method),
                     linewidth=2)

            # Make the initial position and times explicit
            plt.axhline(self.locf[0], color='b', linestyle='--', label='first location fit')
            plt.axvline(self.timef[0], color='b', linestyle=':', label='first time fit')

            # Show the velocity and acceleration (if appropriate)
            plt.text(x_pos[1], y_pos[1], r'v={:f}$\pm${:f}'.format(self.velocity.value, self.velocity_error))
            if self.n_degree > 1:
                plt.text(x_pos[2], y_pos[2], r'a={:f}$\pm${:f}'.format(self.acceleration.value, self.acceleration_error))
        else:
            if not self.fit_able:
                plt.text(x_pos[1], y_pos[1], 'arc not fitable')
            elif not self.fitted:
                plt.text(x_pos[2], y_pos[2], 'arc was fitable, but no fit found')

        # Show the plot
        plt.xlim(0.0, self.times[-1])
        plt.legend(framealpha=0.8)
        plt.show()


class EstimateDerivativeByrne2013:
    """
    An object that calculates the velocity and acceleration of a portion of the
    wavefront.  The calculation is implemented using the Byrne et al 2013, A&A,
    557, A96, 2013 approach.

    Parameters
    ----------
    times : one-dimensional Quantity array of size nt with units convertible
            to seconds

    position : one-dimensional Quantity array of size nt with units convertible
            to degrees of arc

    error : one-dimensional Quantity array of size nt with units convertible
            to degrees of arc

    """

    @u.quantity_input(times=u.s, position=u.degree, error=u.degree)
    def __init__(self, times, position, error, n_trials, window_length, polyorder, **savitsky_golay_kwargs):

        self.times = times.to(u.s).value
        self.position = position.to(u.degree).value
        self.error = error.to(u.degree).value
        self.n_trials = n_trials

        #
        # Byrne et al (2013) use the Savitzky-Golay method to estimate
        # derivatives.
        #
        self.window_length = window_length
        self.polyorder = polyorder
        self.savitsky_golay_kwargs = savitsky_golay_kwargs

        #
        # Byrne et al (2013) use a bootstrap to estimate errors in the
        # derivative.
        #
        i = 0
        while i < n_trials:
            self.svf = savgol_filter(self.position, self.window_length,
                                     self.polyorder, **savitsky_golay_kwargs)

    def peek(self):
        """
        Make a plot of the estimated derivative.
        """
        pass
