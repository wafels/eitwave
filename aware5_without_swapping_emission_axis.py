#
# Demonstration AWARE algorithm
#
# This file contains the image processing, arc definition and model fitting
# routines that comprise the AWARE algorithm.
#
#
from copy import deepcopy
import pickle
import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
from scipy.ndimage.filters import median_filter
from scipy.ndimage import grey_closing
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import scipy.optimize as op
from scipy.interpolate import interp1d
from skimage.morphology import disk
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import astropy.units as u

from sunpy.map import Map
from sunpy.time import parse_time

import aware_utils
from aware_constants import solar_circumference_per_degree_in_km
import mapcube_tools


class Processing:
    def __init__(self, mc, radii=[[11, 11]*u.degree], clip_limit=None,
                 histogram_clip=[0.0, 99.], func=np.sqrt, develop=None):

        self.mc = mc
        self.radii = radii
        self.clip_limit = clip_limit
        self.histogram_clip = histogram_clip
        self.func = func
        self.develop = develop

    def persistence(self):
        return mapcube_tools.persistence(self.mc)

    def running_difference(self, **kwargs):
        return mapcube_tools.running_difference(self.mc, **kwargs)


#
# AWARE:  image processing
#
def _apply_median_filter(nr, footprint, three_d):
    if three_d:
        pancake = np.swapaxes(np.tile(footprint, (3, 1, 1)), 0, -1)
        nr = 1.0*median_filter(nr, footprint=pancake)
    else:
        nt = nr.shape[2]
        for i in range(0, nt):
            nr[:, :, i] = 1.0*median_filter(nr[:, :, i], footprint=footprint)
    return nr


def _apply_closing(nr, footprint, three_d):
    if three_d:
        pancake = np.swapaxes(np.tile(footprint, (3, 1, 1)), 0, -1)
        nr = 1.0*grey_closing(nr, footprint=pancake)
    else:
        nt = nr.shape[2]
        for i in range(0, nt):
            nr[:, :, i] = 1.0*grey_closing(nr[:, :, i], footprint=footprint)
    return nr


def replace_data_in_mapcube(datacube, mc):
    new_mc = []
    for i, m in enumerate(mc):
        new_mc.append(Map(datacube[:, :, i], m.meta))
    return Map(new_mc, cube=True)


@mapcube_tools.mapcube_input
def processing(mc, radii=[[11, 11]*u.degree],
               clip_limit=None,
               histogram_clip=[0.0, 99.],
               func=np.sqrt,
               inv_func=np.square,
               three_d=False,
               returned=None):
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
    three_d :
    returned :

    """
    answer = dict()
    # Define the disks that will be used on all the images.
    # The first disk in each pair is the disk that is used by the median
    # filter.  The second disk is used by the morphological closing
    # operation.
    disks = []
    for r in radii:
        e1 = (r[0]/mc[0].scale.x).to('pixel').value  # median circle radius - across wavefront
        e3 = (r[1]/mc[0].scale.x).to('pixel').value  # closing circle width - across wavefront
        disks.append([disk(e1), disk(e3)])

    # Calculate the persistence
    new = mapcube_tools.persistence(mc)
    if 'persistence' in returned:
        answer['persistence'] = new

    # Calculate the running difference
    new = mapcube_tools.running_difference(new)
    if 'running_difference' in returned:
        answer['running_difference'] = new

    # Storage for the processed mapcube.
    new_mc = []

    # Only want positive differences, so everything lower than zero
    # should be set to zero
    mc_data = func(new.as_array())
    mc_data[mc_data < 0.0] = 0.0
    if 'replace_zeroes' in returned:
        answer['replace_zeroes'] = replace_data_in_mapcube(mc_data, new)

    # Get rid of NaNs
    nans_here = np.logical_not(np.isfinite(mc_data))
    mc_data2 = deepcopy(mc_data)
    mc_data2[nans_here] = 0.0
    if 'replace_nans' in returned:
        answer['replace_nans'] = replace_data_in_mapcube(mc_data2, new)

    # Clip the data to be within a range, and then normalize it.
    if clip_limit is None:
        cl = np.nanpercentile(mc_data2, histogram_clip)
    mc_data2[mc_data2 > cl[1]] = cl[1]
    mc_data2[mc_data2 < cl[0]] = cl[0]
    if 'clipped' in returned:
        answer['clipped'] = replace_data_in_mapcube(inv_func(mc_data2), new)

    # Rescale
    mc_data2 = (mc_data2 - cl[0]) / (cl[1]-cl[0])
    if 'rescaled' in returned:
        answer['rescaled'] = replace_data_in_mapcube(mc_data2, new)

    # Clean the data to isolate the wave front.  Use three dimensional
    # operations from scipy.ndimage.  This approach should get rid of
    # more noise and have better continuity in the time-direction.
    final = np.zeros_like(mc_data2, dtype=np.float32)

    # Do the cleaning and isolation operations on multiple length-scales,
    # and add up the final results.
    nr = deepcopy(mc_data2)
    # Use three-dimensional filters
    for j, d in enumerate(disks):
        pancake = np.swapaxes(np.tile(d[0], (3, 1, 1)), 0, -1)

        print('\n', nr.shape, pancake.shape, '\n', 'started median filter.')
        nr = _apply_median_filter(nr, d[0], three_d)

        print(' started grey closing.')
        nr = _apply_closing(nr, d[1], three_d)

        # Sum all the
        final += nr*1.0

    # Create the list that will be turned in to a mapcube
    print('Creating list that will be turned in to a mapcube.')
    for i, m in enumerate(new):
        new_map = Map(ma.masked_array(final[:, :, i], mask=nans_here[:, :, i]),
                      m.meta)
        new_map.plot_settings = deepcopy(m.plot_settings)
        new_mc.append(new_map)

    # Return the cleaned mapcube
    answer['cleaned'] = Map(new_mc, cube=True)
    return answer


################################################################################
# AWARE: building the latitude - time data
#
def build_lat_time_data(lon, extract, segmented_maps):
    # Build up the data at this longitude
    pixels = extract[lon][0]
    latitude = extract[lon][1]
    nlat = len(latitude)
    nt = len(segmented_maps)

    # Define the array that will hold the emission data along the
    # great arc at all times
    lat_time_data = np.zeros((nlat, nt))
    x = pixels[0, :]
    y = pixels[1, :]
    for t in range(0, nt):
        lat_time_data[:, t] = segmented_maps[t].data[y[:], x[:]]
    return lat_time_data


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
    return amplitude * np.exp(-0.5*onent**2)


def estimate_fwhm(x, y, maximum, arg_maximum):
    half_max = 0.5*maximum
    above = y > half_max
    if np.sum(above) < 3:
        return np.nan
    x_lhs = np.nanmin(x[above])
    x_rhs = np.nanmax(x[above])
    if x_lhs < arg_maximum < x_rhs:
        return x_rhs - x_lhs
    else:
        return np.nan


@mapcube_tools.mapcube_input
def get_times_from_start(mc, start_date=None):
    # Get the times of the images
    if start_date is None:
        start_time = parse_time(mc[0].date)
    else:
        start_time = parse_time(start_date)
    return np.asarray([(parse_time(m.date) - start_time).seconds for m in mc]) * u.s


@u.quantity_input(latitude=u.degree)
def average_position(data, latitude):
    """
    Calculate an average position of the wavefront
    :param data: ndarray of size (nlat, nt)
    :param latitude:
    :return:
    """
    nt = data.shape[1]

    # Average position
    pos = np.zeros(nt)
    for i in range(0, nt):
        emission = data[:, i]
        summed_emission = np.nansum(emission)
        pos[i] = np.nansum(emission * latitude.to(u.degree).value) / summed_emission
    return pos * u.degree


def position_index(data, func=np.nanmin):
    """
    Calculate the minimum position of the wavefront.
    :param data: ndarray of size (nlat, nt)
    :return:
    """
    nt = data.shape[1]

    # Minimum Position
    pos = np.zeros(nt, dtype=int)
    mask = np.zeros(nt, dtype=np.bool)
    for i in range(0, nt):
        emission = data[:, i]
        w = np.where(emission > 0.0)[0]
        if w.size > 0:
            pos[i] = func(w)
            mask[i] = False
        else:
            pos[i] = -9999
            mask[i] = True
    return np.ma.array(pos, mask=mask)


@u.quantity_input(latitude=u.degree)
def this_position(data, latitude, func=np.nanmin):
    """
    Calculate the maximum position of the wavefront
    :param data: ndarray of size (nlat, nt)
    :param latitude:
    :return:
    """
    nt = data.shape[1]
    pos = np.zeros(nt)
    posi = position_index(data, func=func)
    print(posi)
    for i in range(0, nt):
        p = posi[i]
        print(p)
        if np.isfinite(p):
            pos[i] = latitude[p].to(u.deg).value
        else:
            pos[i] = np.nan
    return pos * u.degree


@u.quantity_input(latitude=u.degree)
def central_position(data, latitude):
    """
    Calculate the central position of the wavefront
    :param data: ndarray of size (nlat, nt)
    :param latitude:
    :return:
    """
    return 0.5 * (this_position(data, latitude, func=np.nanmin) +
                  this_position(data, latitude, func=np.nanmax))


@u.quantity_input(latitude=u.degree)
def weighted_central_position(data, latitude):
    """
    Calculate the central position of the wavefront
    :param data: ndarray of size (nlat, nt)
    :param latitude:
    :return:
    """
    nt = data.shape[1]

    # Central positions at all times
    center = central_position(data, latitude).to(u.deg).value

    # Weighted central position
    pos = np.zeros(nt)
    for i in range(0, nt):
        emission = data[:, i]
        summed_emission = np.nansum(emission)
        if np.isfinite(center[i]):
            difference_from_center = latitude.to(u.deg).value - center[i]
            weighted_offset = np.nansum(emission * difference_from_center) / summed_emission
            pos[i] = center[i] + weighted_offset
        else:
            pos[i] = np.nan
    return pos * u.deg


@u.quantity_input(latitude=u.degree)
def position_and_error_by_fitting_gaussian(data, latitude, sigma=None):
    """
    Calculate the position of the wavefront by fitting a Gaussian profile.
    :param data: emission data
    :param latitude: latitudinal extent of the emission data
    :param sigma: estimated error in the data
    :return: an estimate of the position and the error in the position, of the wavefront
    """
    nt = data.shape[1]
    position = np.zeros(nt)
    error = np.zeros_like(position)
    latitude_value = latitude.to(u.degree).value
    for i in range(0, nt):
        emission_data = data.data[:, i]
        emission_mask = data.mask[:, i]
        sigma_input = sigma.data[:, i]
        sigma_input_mask = sigma.mask[:, i]
        fit_here = np.logical_and(np.logical_and(~emission_mask, ~sigma_input_mask), np.isfinite(emission_data))
        if np.sum(fit_here) < 3:
            position[i] = np.nan
            error[i] = np.nan
        else:
            edfh = emission_data[fit_here]
            amplitude_estimate = np.nanmax(edfh)
            position_estimate = latitude_value[np.nanargmax(edfh)]
            fwhm = estimate_fwhm(latitude_value[fit_here],
                                 edfh,
                                 amplitude_estimate,
                                 position_estimate)
            if fwhm is np.nan:
                position[i] = np.nan
                error[i] = np.nan
            else:
                sd_estimate = fwhm/(2*np.sqrt(2*np.log(2.)))
                gaussian_amplitude_estimate = np.nanmax(edfh)
                p0 = np.asarray([gaussian_amplitude_estimate, position_estimate, sd_estimate])
                try:
                    # Since we are fitting the intensity of the wavefront here, an estimate for the error
                    # in the intensity should be supplied.  If purely due to Poisson noise, then we
                    # should use the square root of the intensity of the data from which this wave signal
                    # has been extracted.  That is likely to be large compared to the size of the signal.
                    # A simpler, but very gross underestimate, is to simply take the square root of the
                    # estimated wave signal itself.
                    if sigma is None:
                        sigma_input = np.sqrt(edfh)
                    popt, pcov = curve_fit(gaussian, latitude_value[fit_here], edfh, p0=p0, sigma=sigma_input[fit_here])
                    position[i] = popt[1]
                    error[i] = np.sqrt(np.diag(pcov))[1]
                except RuntimeError:
                    position[i] = np.nan
                    error[i] = np.nan
    return position*u.degree, error*u.degree


@u.quantity_input(latitude=u.degree)
def wavefront_position_error_estimate_standard_deviation(data, latitude):
    """
    Calculate the standard deviation of the width of the wavefornt
    :param data:
    :param latitude:
    :return:
    """
    nt = data.shape[1]
    error = np.zeros(nt)
    for i in range(0, nt):
        emission = data[:, i]
        summed_emission = np.nansum(emission)
        try:
            error[i] = np.nanstd(emission*latitude.to(u.degree).value) / np.sqrt(summed_emission)
        except TypeError:
            error[i] = np.nan
    return error * u.degree


@u.quantity_input(lat_bin=u.degree/u.pix)
def wavefront_position_error_estimate_width(data, lat_bin, position_choice='maximum'):
    """
    Calculate the standard deviation of the width of the wavefornt
    :param data:
    :param lat_bin:
    :param position_choice:
    :return:
    """
    single_pixel_std = np.sqrt(1.0 / 12.0)
    nt = data.shape[1]
    error = np.zeros(nt)
    for i in range(0, nt):
        emission = deepcopy(data[:, i])
        # Find where the emission is NaN
        nan_emission = np.where(np.isnan(emission))[0]
        # Set the NaN emission to zero.  This is because the width estimate
        # is intended to examine the non-zero, non-NaN emission.
        emission[nan_emission[:]] = 0
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
    @u.quantity_input(latitude=u.degree, longitude=u.degree)
    def __init__(self, data, times, latitude, longitude, start_time=None, sigma=None, title=None):
        """
        Object to store data on the emission along each arc as a function of
        time.

        data : ndarray of size (nlat, nt)
            emission along a line of latitude as a function of time

        times : list-like
            sunpy time objects corresponding to the time the emission was
            seen

        latitude : ndarray of size nlat
            latitude bins of the arc

        Keywords
        --------
        start_time : sunpy time object
            the initial time that the event is thought to have started at.

        sigma : ndarray of size (nlat, nt)
            an estimate of the uncertainty in the emission

        title : `str`
            a description of this arc.

        """

        self.data = data
        self.times = times
        self.latitude = latitude
        self.lat_bin = (self.latitude[1] - self.latitude[0])/u.pix
        self.longitude = longitude
        if start_time is None:
            self.start_time = times[0]
        else:
            self.start_time = start_time
        self.sigma = sigma
        if title is None:
            self.title = 'longitude=%s' % str(self.longitude)
        else:
            self.title = title

        # Time in seconds since the start time
        self.t = np.asarray([(parse_time(time) - self.start_time).seconds for time in self.times]) * u.s

        # How to measure the position of the wavefront along the arc as a
        # function of time along the arc.
        #self.position_choice = None

        # The position of the wavefront along the arc as a function of time.
        #self.position = None

        # How to measure the error in position of the wavefront.
        #self.error_choice = None

        # The error in the position of the wavefront long the arc as a function
        # of time.
        #self.position_error = None

    def average_position(self):
        return average_position(self.data, self.latitude)

    def maximum_position(self):
        return maximum_position(self.data, self.latitude)

    def position_and_error_by_fitting_gaussian(self):
        return position_and_error_by_fitting_gaussian(self.data, self.latitude, sigma=self.sigma)

    def wavefront_position_error_estimate_standard_deviation(self):
        return wavefront_position_error_estimate_standard_deviation(self.data, self.latitude)

    def wavefront_position_error_estimate_width(self, position_choice):
        return wavefront_position_error_estimate_width(self.data, self.lat_bin, position_choice=position_choice)

    def weighted_central_position(self):
        return weighted_central_position(self.data, self.latitude)

    def locator(self, position_choice, error_choice):

        if error_choice == 'std':
            position_error = self.wavefront_position_error_estimate_standard_deviation()
        elif error_choice == 'width':
            position_error = self.wavefront_position_error_estimate_width(position_choice)
        else:
            raise ValueError('Unrecognized error choice.')

        if position_choice == 'average':
            position = self.average_position()
        elif position_choice == 'maximum':
            position = self.maximum_position()
        elif position_choice == 'gaussian':
            position, position_error = self.position_and_error_by_fitting_gaussian()
        elif position_choice == 'weighted_center':
            position = self.weighted_central_position()
        else:
            raise ValueError('Unrecognized position choice.')

        return position, position_error

    def peek(self):
        plt.imshow(self.data, aspect='auto', interpolation='none',
                   origin='lower',
                   extent=[self.t[0].to(u.s).value,
                           self.t[-1].to(u.s).value,
                           self.latitude[0].to(u.degree).value,
                           self.latitude[-1].to(u.degree).value])
        plt.xlim(0, self.t[-1].to(u.s).value)
        if self.t[0].to(u.s).value > 0.0:
            plt.fill_betweenx([self.latitude[0].to(u.degree).value,
                               self.latitude[-1].to(u.degree).value],
                              self.t[0].to(u.s).value,
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


def forward_diff(x):
    return x[1:] - x[0:-1]


def strictly_monotonic_increasing(x):
    return np.all(forward_diff(x) > 0)


def monotonic_increasing(x):
    return np.all(forward_diff(x) >= 0)


def turning_point(a, b):
    return -b / (2*a)


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
    t : one-dimensional ndarray Quantity array of size nt
        seconds since the initial time

    position : one-dimensional ndarray Quantity array of size nt
        location of the wavefront since the initial time

    error : one-dimensional ndarray Quantity array of size nt
        error in the location of the wavefront since the initial time

    n_degree : int
        degree of the polynomial to fit

    ransac_kwargs : dict
        keywords for the RANSAC algorithm

    arc_identity : `object`
        a free-form holder for identity information for this arc.

    error_tolerance_kwargs : dict
        keywords controlling which positions have a tolerable about of error.
        Only positions that satisfy these conditions go on to be fit.

    """

    @u.quantity_input(t=u.s, position=u.degree, error=u.degree)
    def __init__(self, t, position, error, n_degree=2, ransac_kwargs=None,
                 error_tolerance_kwargs=None, arc_identity=None,
                 fit_method='poly_fit', constrained_fit_method='L-BFGS-B',
                 cvt_factor=2.0, lasso_alpha=0.01):

        self.t = t.to(u.s).value
        self.nt = len(t)
        self.position = position.to(u.degree).value
        self.error = error.to(u.degree).value
        self.n_degree = n_degree
        self.ransac_kwargs = ransac_kwargs
        self.error_tolerance_kwargs = error_tolerance_kwargs
        self.arc_identity = arc_identity
        self.fit_method = fit_method
        self.constrained_fit_method = constrained_fit_method
        self.cvt_factor = cvt_factor

        # LASSO variables
        self.lasso_alpha = lasso_alpha

        # At the outset, assume that the arc is able to be fit.
        self.fit_able = True

        # Has the arc been fitted?
        self.fitted = None

        # Fit report
        self.fit_report = "Fit report"

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
                this_x = deepcopy(self.t[self.defined])
                this_y = deepcopy(self.position[self.defined])

                # Try the RANSAC fit
                self.ransac_residual_error = np.median(
                    self.error[self.defined])
                self.ransac_model = make_pipeline(
                    PolynomialFeatures(self.n_degree), RANSACRegressor(
                        residual_threshold=self.ransac_residual_error))
                try:
                    self.ransac_model.fit(this_x.reshape((len(this_x), 1)), this_y)
                    self.inlier_mask = np.asarray(self.ransac_model.named_steps['ransacregressor'].inlier_mask_)
                    self.ransac_success = True
                except ValueError:
                    self.inlier_mask = np.ones(np.sum(self.defined), dtype=bool)
                    self.ransac_success = False

                # Try the LASSO fit
                self.model_lasso = make_pipeline(PolynomialFeatures(self.n_degree), Lasso(alpha=self.lasso_alpha, normalize=True))
                try:
                    self.model_lasso.fit(this_x.reshape((len(this_x), 1)), this_y.reshape((len(this_x), 1)))
                    self.lasso_success = True
                except ValueError:
                    self.lasso_success = False
            else:
                self.ransac_success = None
                self.inlier_mask = np.ones(np.sum(self.defined), dtype=bool)
        else:
            self.ransac_success = None
            self.inlier_mask = np.ones(np.sum(self.defined), dtype=bool)

        # Are there enough points to do a fit?
        if np.sum(self.inlier_mask) <= 3:
            self.fit_able = False
            self.fitted = False
            self.fit_report += ': 3 or less points available to fit'

        # Perform a fit if there enough points
        if self.fit_able:
            #
            self.fit_report += ': arc has sufficient number of points to be fitable'

            # Get the locations where the location is defined
            self.locf = self.position[self.defined][self.inlier_mask]

            # Get the error where the location is defined
            self.errorf = self.error[self.defined][self.inlier_mask]

            # Determine the indices that are fitable
            self.indicesf = (np.arange(self.nt))[self.defined][self.inlier_mask]

            # Errors which are too small can really bias the fit.  The entire
            # fit can be pulled to take into account a couple of really bad
            # points.  This section attempts to fix that by giving those points
            # a user-defined value.
            if self.error_tolerance_kwargs is not None:
                if 'threshold_error' in self.error_tolerance_kwargs.keys():
                    self.threshold_error = self.error_tolerance_kwargs['threshold_error'](self.error[self.defined])
                    if 'function_error' in self.error_tolerance_kwargs.keys():
                        self.errorf[self.errorf < self.threshold_error] = self.error_tolerance_kwargs['function_error'](self.error[self.defined])
                    else:
                        self.errorf[self.errorf < self.threshold_error] = self.threshold_error

            # Get the times where the location is defined
            self.timef = self.t[self.defined][self.inlier_mask]

            # Do the fit to the data
            try:
                # Where the velocity will be stored in the final results
                self.vel_index = self.n_degree - 1

                #
                # Polynomial and conditional fits to the data
                #
                if self.fit_method == 'poly_fit' or self.fit_method == 'conditional':
                    #
                    # Should try a different fitting routine since it since this
                    # one seems to behave well only with small errors
                    #
                    self.estimate, self.covariance = np.polyfit(self.timef, self.locf, self.n_degree, w=1.0/self.errorf, cov=True)

                    # If the code gets this far, then we can assume that a fit
                    # has completed
                    self.fitted = True

                    # Best fit
                    self.best_fit = np.polyval(self.estimate, self.timef)

                    # Estimated error
                    self.estimate_error = np.abs(np.sqrt(np.diag(self.covariance)))

                    # Calculate the error in the resulting best fits
                    nerrors = 2**(self.n_degree + 1)
                    self.best_fit_error = np.zeros((nerrors, len(self.best_fit)))
                    for i in range(0, nerrors):
                        binary = 2*(np.asarray([int(s) for s in np.binary_repr(i, width=self.n_degree + 1)]) - 0.5)
                        self.best_fit_error[i, :] = np.polyval(self.estimate + binary*self.estimate_error, self.timef)

                    # Error in velocity
                    ve = self.estimate_error[self.vel_index]

                    # Calculate the conditional velocity trigger
                    self.conditional_velocity_trigger = self.estimate[self.vel_index] + self.cvt_factor * ve
                    if self.fit_method == 'conditional' and self.conditional_velocity_trigger < 0:
                        self.constrained_minimization()
                        self.fit_method = 'conditional (constrained)'
                #
                # Constrained fit to the data
                #
                if self.fit_method == 'constrained':
                    self.constrained_minimization()

                # Assume uniform wavefronts
                if self.fit_method == 'assume_uniform_wavefronts':
                    self.assume_uniform_wavefronts()

                # Convert to deg/s
                self.velocity = self.estimate[self.vel_index] * u.deg/u.s
                self.velocity_error = np.sqrt(self.covariance[self.vel_index, self.vel_index]) * u.deg/u.s
                self.s0_extrapolated = self.locf[0] * u.deg - self.velocity * self.timef[0] * u.s

                # Convert to deg/s/s
                if self.n_degree >= 2:
                    self.acc_index = self.n_degree - 2
                    self.acceleration = 2 * self.estimate[self.acc_index] * u.deg/u.s/u.s
                    self.acceleration_error = 2 * np.sqrt(self.covariance[self.acc_index, self.acc_index]) * u.deg/u.s/u.s
                    # Extrapolated velocity at the first time.
                    self.v0_extrapolated = self.velocity - (self.acceleration * self.timef[0]) * u.s
                    self.s0_extrapolated = self.locf[0] * u.deg -\
                                           self.velocity * self.timef[0] * u.s -\
                                           0.5 * self.acceleration * (self.timef[0]*u.s) ** 2
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
                                                        self.nt,
                                                        self.indicesf,
                                                        velocity_error=self.velocity_error,
                                                        acceleration_error=self.acceleration_error)

                # The fraction of the input arc was actually used in the fit
                self.arc_duration_fraction = len(self.timef) / (1.0 * self.nt)

                # Waves can't go backwards
                if self.n_degree == 2:
                    self.monotonic_increasing = monotonic_increasing(self.best_fit)
                    if not self.monotonic_increasing:
                        self.fit_report += ": best fit arc not monotonic increasing in fit time range"
                        self.fitted = False

            except LA.LinAlgError:
                # Error in the fitting algorithm
                self.fit_report += ": LA.LinAlgError in fitting"
                self.fitted = False
            except ValueError:
                # Error in the fitting algorithm
                self.fit_report += ": ValueError in fitting"
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

    def assume_uniform_wavefronts(self):
        """
        This fitting method assumes that the width of the wavefronts can be
        treated as a uniform block of intensity and it is not possible to
        locate the peak of the wavefront other than to say it is somwehere in
        here.  Therefore, the location of the wavefront is uniformly
        distributed.

        A fit and error are generated as follows.  Assume that y_{true} is in
        the range [y_{i} - \sigma_{i}, y_{i} + \sigma_{i}].  Assume a uniform
        distribution in this range and pick y'_{i}.  Fit this assuming the
        uniform distribution at each point.  Repeat this process M times.  The
        final result is the mean and standard distribution of all the
        polynomial coefficients.

        """
        model_function = np.polyval
        auw_method = 'Nelder-Mead'
        ntrials = 100
        n_locf = len(self.locf)
        storage = np.zeros((ntrials, self.n_degree + 1))

        for ntrial in range(0, ntrials):
            # Get the next sample
            ydash = np.zeros(n_locf)
            for i in range(0, n_locf):
                ydash[i] = np.random.uniform(low=self.locf[i]-self.errorf[i],
                                             high=self.locf[i]+self.errorf[i])
            initial_guess = np.polyfit(self.timef, ydash, self.n_degree, w=1.0/(self.errorf ** 2))

            # Fit the next sample
            auw_fit = self.auw_go(self.timef, ydash, model_function, initial_guess, auw_method)
            self.fitted = auw_fit['success']
            storage[ntrial, :] = auw_fit['x']

        # Store the results
        self.estimate = np.mean(storage, axis=0)
        self.covariance = np.diag(np.std(storage, axis=0)) # Error estimate
        self.best_fit = model_function(self.estimate, self.timef)

    #
    # Log likelihood function.  In this case we want the product of normal
    # distributions.
    #
    def auw_lnlike(self, variables, x, y, model_function):
        """
        Log likelihood of the data given a model.  Assumes that the data is
        uniformly distributed.
        :param variables: array like, variables used by model_function
        :param x: the independent variable (most often normalized frequency)
        :param y: the dependent variable (observed power spectrum)
        :param model_function: the model that we are using to fit the power
        spectrum
        :return: the log likelihood of the data given the model.
        """
        model = model_function(variables, x)
        exceed_above = model >= self.locf + self.errorf
        exceed_below = model <= self.locf - self.errorf
        if np.any(exceed_above) or np.any(exceed_below):
            return -np.inf
        else:
            return 0.0

    #
    # Fit the input model to the data.
    #
    def auw_go(self, x, data, model_function, initial_guess, method):
        nll = lambda *args: -self.auw_lnlike(*args)
        args = (x, data, model_function)
        return op.minimize(nll, initial_guess, args=args, method=method)

    def get_interpolated(self, nt=None):
        """
        Return sample times and interpolated data in the time range where the data
        was fitted.  This interpolated data can be used to estimate the velocity
        and acceleration of the wave along an arc using the Savitzky-Golay filtering
        method of Byrne et al (2013).

        :return: tuple
            The first element is the sample times.  The second element is the
            interpolated data.
        """
        f = interp1d(self.timef, self.locf)
        duration = self.timef[-1] - self.timef[0]
        if nt is None:
            dt = 12.0*u.s
            new_nt = 1 + int(duration // dt.to(u.s).value)
        else:
            if not isinstance(nt, int):
                print('Input is not an integer - attempting to cast to int')
                new_nt = int(nt)
            else:
                new_nt = nt
            dt = (duration / (new_nt-1))*u.s
        new_timef = self.timef[0].to(u.s) + dt*np.arange(0, new_nt)
        return new_timef, f(new_timef)*u.deg

    def plot(self, title=None, zero_at_start=False, savefig=None, figsize=(8, 6), line=None, fontsize=20, legend_fontsize=12):
        """
        A summary plot of the results the fit.
        """
        # Fontsizes
        xy_tick_label_factor = 0.8

        # String formats and nice formatting for values
        v_format = '{:.0f}'
        ve_format = '{:.0f}'
        vel_string = r' km s$^{-1}$'

        a_format = '{:.3f}'
        ae_format = '{:.3f}'
        acc_string = r' km s$^{-2}$'

        # Initial value
        if zero_at_start:
            offset = np.nanmin(self.position)
        else:
            offset = 0.0

        # Calculate positions for plotting text
        ny_pos = 3
        y_pos = np.zeros(ny_pos)
        for i in range(0, ny_pos):
            y_min = np.nanmin(self.position - self.error + offset)
            y_max = np.nanmax(self.position + self.error + offset)/2.0
            y_pos[i] = y_min + i * (y_max - y_min) / (1.0 + 1.0*ny_pos)
        x_pos = np.zeros_like(y_pos)
        x_pos[:] = np.nanmin(self.t) + 0.5*(np.nanmax(self.t) - np.nanmin(self.t))

        # Show all the data
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.errorbar(self.t, self.position + offset, yerr=self.error,
                    color='k', label='all data')

        # Information labels
        ax.set_xlabel('time (seconds) [{:n} images]'.format(len(self.t)), fontsize=fontsize)
        ax.set_ylabel('degrees of arc from initial position', fontsize=fontsize)
        if title is None:
            if self.arc_identity is not None:
                title = '{:.0f}'.format(self.arc_identity.value) + 'deg'
            else:
                title = None
            ax.set_title(title, fontsize=fontsize)
        else:
            ax.set_title(title, fontsize=fontsize)
        ax.text(x_pos[0], y_pos[0], 'polynomial degree = {:n}'.format(self.n_degree),
                fontsize=fontsize, bbox=dict(facecolor='y', alpha=0.5))

        # Show areas where the position is not defined
        at_least_one_not_defined = False
        for i in range(0, self.nt):
            if not self.defined[i]:
                if i == 0:
                    t0 = self.t[0]
                    t1 = 0.5*(self.t[i] + self.t[i+1])
                elif i == self.nt-1:
                    t0 = 0.5*(self.t[i-1] + self.t[i])
                    t1 = self.t[self.nt-1]
                else:
                    t0 = 0.5*(self.t[i-1] + self.t[i])
                    t1 = 0.5*(self.t[i] + self.t[i+1])
                if not at_least_one_not_defined:
                    at_least_one_not_defined = True
                    ax.axvspan(t0, t1, color='b', alpha=0.1, edgecolor='none', label='no detection')
                else:
                    ax.axvspan(t0, t1, color='b', alpha=0.1, edgecolor='none')

        if self.fitted:

            # Show the data used in the fit
            ax.errorbar(self.timef, self.locf + offset, yerr=self.errorf,
                        marker='o', linestyle='None', color='r',
                        label='data used in fit')

            # Show the best fit arc
            ax.plot(self.timef, self.best_fit + offset, color='r', label='best fit ({:s})'.format(self.fit_method),
                    linewidth=2)

            # Make the initial position and times explicit
            ax.axhline(self.locf[0] + offset, color='b', linestyle='--', label='first location fit')
            ax.axvline(self.timef[0], color='b', linestyle=':', label='first time fit (t={:n}s)'.format(self.timef[0]))

            # Show the velocity and acceleration (if appropriate)
            velocity_string = r'v=' +\
                              v_format.format((solar_circumference_per_degree_in_km*self.velocity).value) +\
                              '$\pm$' +\
                              ve_format.format((solar_circumference_per_degree_in_km*self.velocity_error).value) +\
                              vel_string
            ax.text(x_pos[1], y_pos[1], velocity_string,
                    fontsize=fontsize, bbox=dict(facecolor='y', alpha=0.5))
            if self.n_degree > 1:
                acceleration_string = r'a=' +\
                                      a_format.format((solar_circumference_per_degree_in_km*self.acceleration).value) +\
                                      '$\pm$' +\
                                      ae_format.format((solar_circumference_per_degree_in_km*self.acceleration_error).value) +\
                                      acc_string
                ax.text(x_pos[2], y_pos[2], acceleration_string,
                        fontsize=fontsize, bbox=dict(facecolor='y', alpha=0.5))
        else:
            if not self.fit_able:
                ax.text(x_pos[1], y_pos[1], 'arc not fitable',
                        fontsize=fontsize, bbox=dict(facecolor='y', alpha=0.5))
            elif not self.fitted:
                ax.text(x_pos[2], y_pos[2], 'arc was fitable, but no fit found',
                        fontsize=fontsize, bbox=dict(facecolor='y', alpha=0.5))

        # Increase the size of the x and y tick labels
        xtl = ax.axes.xaxis.get_majorticklabels()
        for l in range(0, len(xtl)):
            xtl[l].set_fontsize(xy_tick_label_factor*fontsize)
        ytl = ax.axes.yaxis.get_majorticklabels()
        for l in range(0, len(ytl)):
            ytl[l].set_fontsize(xy_tick_label_factor*fontsize)

        # Extra line
        if line is not None:
            ax.plot(line['t'], line['y'], **line['kwargs'])

        # Show the plot
        ax.grid('on', linestyle=':')
        ax.set_xlim(0.0, self.t[-1])
        ax.legend(framealpha=0.5, loc=2, fontsize=legend_fontsize, facecolor='yellow')
        #fig.tight_layout()
        if savefig is not None:
            fig.savefig(savefig)
        return fig, ax


class EstimateDerivativesByrne2013:
    @u.quantity_input(t=u.s, position=u.degree)
    def __init__(self, t, position, window_length, polyorder, delta=12.0*u.s, **savitsky_golay_kwargs):
        """
        An object that estimates the position, velocity and acceleration of a
        portion of the wavefront using a bootstrap and Savitsky-Golay filter.
        See the reference below for more details.

        Parameters
        ----------
        t : one-dimensional ndarray Quantity array of size nt
            seconds since the initial time

        position : one-dimensional ndarray Quantity array of size nt
            location of the wavefront since the initial time

        window_length :

        polyorder :

        Keywords
        --------
        savitsky_golay_kwargs :

        Reference
        ---------
        The calculation is implemented using the approach of Byrne et al
        2013, A&A, 557, A96, 2013.
        """
        self.t = t.to(u.s).value
        self.position = position.to(u.degree).value
        self.nt = len(self.t)

        # Bootstrap defaults
        self.labels = ['position', 'velocity', 'acceleration']
        self.n_bootstrap = 10000
        self.bootstrap_results = dict()
        self.bootstrap_statistics_results = dict()
        for label in self.labels:
            self.bootstrap_results[label] = np.zeros((self.n_bootstrap, self.nt))
            self.bootstrap_statistics_results[label] = dict()

        # Byrne et al (2013) use the Savitzky-Golay method to estimate
        # derivatives.
        self.window_length = window_length
        self.polyorder = polyorder
        self.delta = delta.to(u.s).value
        self.savitsky_golay_kwargs = savitsky_golay_kwargs

        # Calculate an initial Savitsky-Golay estimate of the input data
        self.initial_savgol = savgol_filter(self.position,
                                            self.window_length,
                                            self.polyorder,
                                            deriv=0,
                                            delta=self.delta,
                                            **self.savitsky_golay_kwargs)
        self.error_savgol = self.position - self.initial_savgol

        # Use a bootstrap to calculate errors in position, velocity and
        # acceleration. The results are stored in a  dictionary with three keys
        # - 'position', 'velocity' and 'acceleration'. The key indicates the
        # parameter estimated. Byrne et al (2013) use a bootstrap to estimate
        # errors in the Savitsky-Golay derived kinematics.
        i = 0
        while i < self.n_bootstrap:
            # Bootstrap error with replacement
            bootstrap_error = self.error_savgol[np.random.randint(0, high=self.nt, size=self.nt)]
            for j, label in enumerate(self.labels):
                self.bootstrap_results[label][i, :] = savgol_filter(self.initial_savgol + bootstrap_error,
                                                                    self.window_length,
                                                                    self.polyorder,
                                                                    deriv=j,
                                                                    delta=self.delta,
                                                                    **self.savitsky_golay_kwargs)
            i += 1

        # Return results with the correct dimensions
        for j, label in enumerate(self.labels):
            if j == 0:
                self.bootstrap_results[label] = self.bootstrap_results[label] * u.deg
            else:
                self.bootstrap_results[label] = self.bootstrap_results[label] * u.deg * u.s**-j

        # Calculates statistics based on the bootstrap results. A nested
        # dictionary with three keys at the top level- 'position', 'velocity'
        # and 'acceleration'. The key indicates the parameter estimated.
        # Further dictionary keys at the next level indicate the statistic
        # estimated.

        for label in self.labels:
            results = self.bootstrap_results[label]
            self.bootstrap_statistics_results[label]['median'] = np.median(results, axis=0)
            self.bootstrap_statistics_results[label]['iq_lo'] = np.percentile(results, 25.0, axis=0)
            self.bootstrap_statistics_results[label]['iq_hi'] = np.percentile(results, 75.0, axis=0)
            iqr = self.bootstrap_statistics_results[label]['iq_hi'] - self.bootstrap_statistics_results[label]['iq_lo']
            self.bootstrap_statistics_results[label]['fence_lo'] = self.bootstrap_statistics_results[label]['iq_lo'] - 1.5*iqr
            self.bootstrap_statistics_results[label]['fence_hi'] = self.bootstrap_statistics_results[label]['iq_hi'] + 1.5*iqr

    def peek(self, time_axis=None, title=None, fontsize=20):
        """
        Make a plot of the kinematics.

        Parameters
        ----------
        time_axis :

        title :

        Returns
        -------

        """

        fig, ax = plt.subplots(3, sharex=True)
        ax[0].set_title(title)
        if time_axis is None:
            t = self.t
            ax[2].set_xlabel("time (s)")
        elif len(time_axis) == 1:
            t = self.t
            ax[2].set_xlabel("time (s) from start [{:s}]".format(time_axis[0]))
        else:
            t = time_axis
            fmt = mdates.DateFormatter('%H:%M:%S')
            ax[2].xaxis.set_major_formatter(fmt)
            ax[2].set_xlabel("start time {:%Y/%m/%d %H:%M:%S} UT".format(t[0]))
        ax[2].xaxis.label.set_fontsize(fontsize)

        for j, label in enumerate(self.labels):
            stats = self.bootstrap_statistics_results[label]
            ax[j].set_ylabel(label + '\n(' + stats['median'].unit.to_string('latex_inline') + ')')
            ax[j].yaxis.label.set_fontsize(fontsize)
            ax[j].plot(t, stats['median'], label='median', color='k')
            ax[j].plot(t, stats['iq_lo'], label='interquartile range 25%-75%', color='b', linestyle='--')
            ax[j].plot(t, stats['iq_hi'], color='b', linestyle='--')
            ax[j].plot(t, stats['fence_lo'], label='fence', color='r', linestyle=':')
            ax[j].plot(t, stats['fence_hi'], color='r', linestyle=':')
            if j == 0:
                ax[j].legend(framealpha=0.6, fontsize=fontsize*0.7)
        fig.tight_layout(pad=0)
        plt.show()
