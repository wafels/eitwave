#
# Demonstration AWARE algorithm
#
import os
from copy import deepcopy
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk
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
def processing(mc, radii=[[11, 11]], spike_level=25, accum=1, develop=False):
    """
    Image processing steps used to isolate the EUV wave from the data.  Use
    this part of AWARE to perform the image processing steps that segment
    propagating features that brighten new pixels as they propagate.

    Parameters
    ----------

    mc : sunpy.map.MapCube
    radii : list of lists. Each list contains a pair of numbers that describe the
    radius of the median filter and the closing operation
    spike_level :
    accum :
    """

    # Define the disks that will be used on all the images.
    # The first disk in each pair is the disk that is used by the median
    # filter.  The second disk is used by the morphological closing
    # operation.
    disks = []
    for r in radii:
        disks.append([disk(r[0]), disk(r[1])])

    # For the dump images
    rstring = ''
    for r in radii:
        z = '%i_%i__' % (r[0], r[1])
        rstring += z

    # Calculate the persistence
    new = mapcube_tools.persistence(mc)
    if develop:
        dump_images(new, rstring, '%s_1_persistence' % rstring)

    # Calculate the running difference
    new = mapcube_tools.running_difference(new)
    if develop:
        dump_images(new, rstring, '%s_2_rdiff' % rstring)

    # Storage for the processed data.
    newmc = []
    for im, m in enumerate(new):
        # Dump images - identities
        ident = (rstring, im)

        # Get rid of everything below zero
        newdata = np.clip(m.data, 0.0, np.max(m.data))
        if develop:
            dump_image(newdata, rstring, '%s_3_clipltzero_%05d.png' % ident)

        # Get the square root
        newdata = np.sqrt(newdata)
        if develop:
            dump_image(newdata, rstring, '%s_4_sqrt_%05d.png' % ident)

        # Get rid of spikes
        newdata = np.clip(newdata, np.min(newdata), spike_level * accum)
        if develop:
            dump_image(newdata, rstring, '%s_5_clipspikes_%05d.png' % ident)

        # Isolate the wavefront.
        # First step is to apply a median filter.  This median filter
        # requires that the data be scaled between 0 and 1.
        newdata = newdata / np.max(newdata)
        results = []
        for id, d in enumerate(disks):
            # Get rid of noise by applying the median filter.
            newd = median(newdata, d[0])
            if develop:
                dump_image(newd, rstring, '%s_6_median_%i_%05d.png' % (rstring, radii[id][0], im))

            # Apply the morphological closing operation to rejoin separated
            # parts of the wave front.
            newd = closing(newd, d[1])
            if develop:
                dump_image(newd, rstring, '%s_7_closing_%i_%05d.png' % (rstring, radii[id][1], im))

            results.append(newd)

        if develop:
            dump_image(sum(results), rstring, '%s_final_%05d.png' % ident)
        # New mapcube list
        newmc.append(Map(sum(results), m.meta))

    # Return the cleaned mapcube
    return Map(newmc, cube=True)


def _get_times_from_start(mc):
    # Get the times of the images
    start_time = parse_time(mc[0].date)
    return np.asarray([(parse_time(m.date) - start_time).seconds for m in mc])


def dynamics(unraveled, params,
             originating_event_time=None,
             error_choice='std',
             position_choice='average',
             returned=['arc', 'answer'],
             ransac_kwargs=None):
    """
    Measurement of the progress of the wave across the disk.  This part of
    AWARE generates information concerning the dynamics of the wavefront.
    """
    # Size of the latitudinal bin
    lat_bin = params.get('lat_bin').to('degree').value
    # Get the data
    data = unraveled.as_array()
    # Times
    times = _get_times_from_start(unraveled)
    # Time of the originating event
    if originating_event_time is None:
        originating_event_time = unraveled[0].date
    # Displacement between the time of the originating event and the first
    # measurement.
    offset = (parse_time(unraveled[0].date) - parse_time(originating_event_time)).seconds

    # At all times get an average location of the wavefront
    nlon = data.shape[1]
    nlat = data.shape[0]
    latitude = np.min(unraveled[0].yrange.value) + np.arange(0, nlat) * lat_bin

    results = []
    for lon in range(0, nlon):
        print 'Fitting %i out of %i' % (lon, nlon)
        arc = Arc(data[:, lon, :], times, latitude, offset)
        answer = FitPosition(arc,
                             error_choice=error_choice,
                             position_choice=position_choice,
                             ransac_kwargs=ransac_kwargs)
        # Store the collated results
        z = []
        if 'arc' in returned:
            z.append(arc)
        if 'answer' in returned:
            z.append(answer)
        results.append(z)

    return results


class Arc:
    """
    Object to store data on the emission along each arc as a function of time
    data : ndarray of size (nlat, nt)
    times : ndarray of time in seconds from zero
    latitude : ndarray of the latitude bins of the unraveled array
    """
    def __init__(self, data, times, latitude, offset, title=''):
        self.data = data
        self.offset = offset
        self.times = (times + self.offset) * u.s
        self.nt = times.size
        self.nlat = latitude.size
        self.lat_bin = latitude[1] - latitude[0]
        self.latitude = np.arange(0, self.nlat) * self.lat_bin
        self.title = title

    def peek(self):
        return aware_plot.arc(self)


class FitPosition:
    """
    An object that performs a fit to an AWARE Arc object, and holds the full
    details on how the fit was performed and its results.

    Parameters
    ----------
    arc : Arc
        An AWARE Arc object

    error_choice : {'std' | 'maxwidth'}
        Select which estimate of the error in the position of the wave to use.

    position_choice : {'average' | 'maximum'}
        Select which estimate of the position of the wave to use.

    """

    def __init__(self, arc, error_choice='std', position_choice='average',
                 ransac_kwargs=None):
        # Is the arc fit-able?  Assume that it is.
        self.fit_able = True

        # Has the arc been fitted?
        self.fitted = False

        # Temporal offset - difference between the originating event time and
        # the first possible measurement of the arc.
        self.offset = arc.offset

        # Which error measurement of the position to use when determining the
        # wave.
        self.error_choice = error_choice

        # Which position to use when determining the wave
        self.position_choice = position_choice

        # Use RANSAC to better find inliers?
        self.ransac_kwargs = ransac_kwargs

        # Average position of the remaining emission at each time
        self.av_pos = np.zeros([arc.nt])

        # Position of the wave maximum at each time
        self.maximum = np.zeros_like(self.av_pos)

        # Standard deviation of the remaining emission at each time
        self.std = np.zeros_like(self.av_pos)

        # Error
        self.error = np.zeros_like(self.av_pos)

        # If a location is determined using its pixel location (for example,
        # finding out pixel has the maximum value), then we assume that there is
        # a uniform chance that the actual location is somewhere inside that
        # pixel.  The square of the standard deviation of a uniform
        # distribution is 1/12.
        self._single_pixel_std = arc.lat_bin * np.sqrt(1.0 / 12.0)

        # It has been found through simulation that the first element is
        # on average, over-estimated.  This systematic error reduces the
        # fit velocity and increases the fit acceleration.  We compensate for
        # this by defining a mask that blocks out the first element.
        self._first_position_mask = np.ones_like(self.av_pos, dtype=np.bool)
        #self._first_position_mask[0] = False

        # Maximum extent of remaining emission as measured by subtracting
        # the emission closest to the start from the emission furthest from the
        # start
        self.maxwidth = np.zeros_like(self.av_pos)
        for i in range(0, arc.nt):
            emission = arc.data[::-1, i]
            summed_emission = np.sum(emission)
            nonzero_emission = np.nonzero(emission)
            self.av_pos[i] = np.sum(emission * arc.latitude) / summed_emission
            self.std[i] = np.std(emission * arc.latitude) / summed_emission
            self.maximum[i] = arc.latitude[np.argmax(emission)]
            if len(nonzero_emission[0]) > 0:
                self.maxwidth[i] = arc.lat_bin * (1 + nonzero_emission[0][-1] - nonzero_emission[0][0])
            else:
                self.maxwidth[i] = 0.0

        # Wave sample times
        self.times = arc.times

        # Choose which characterization of the wavefront to use
        if self.position_choice == 'average':
            self.pos = self.av_pos
        if self.position_choice == 'maximum':
            self.pos = self.maximum

        # Locations of the finite data
        self.pos_is_finite = np.isfinite(self.pos)

        # There is at least one location that has emission greater than zero
        self.at_least_one_nonzero_location = np.any(self.pos[self.pos_is_finite] > 0.0)

        # Error choice
        if self.error_choice == 'std':
            self.error = self.std
        if self.error_choice == 'maxwidth':
            for i in range(0, arc.nt):
                if self.maxwidth[i] > 0:
                    if self.position_choice == 'maximum':
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
                    self.error[i] = np.sqrt(self.maxwidth[i] ** 2 +
                                            single_pixel_factor * self._single_pixel_std ** 2)

        # Find if we have enough points to do a quadratic fit
        self.error_is_finite = np.isfinite(self.error)
        self.error_is_above_zero = self.error > 0
        self.defined = self.pos_is_finite * self.error_is_finite * \
                       self.error_is_above_zero * \
                       self.at_least_one_nonzero_location * \
                       self._first_position_mask

        if self.ransac_kwargs is not None:
            # Find inliers using RANSAC, if there are enough points
            if np.sum(self.defined) > 3:
                this_x = deepcopy(self.times[self.defined].value)
                this_y = deepcopy(self.pos[self.defined])
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
        if np.sum(self.defined[self.inlier_mask]) <= 3:
            self.fit_able = False

        # Perform a fit if there enough points
        if self.fit_able:
            # Get the locations where the location is defined
            self.locf = self.pos[self.defined][self.inlier_mask]
            # Get the standard deviation where the location is defined
            self.errorf = self.error[self.defined][self.inlier_mask]
            # Get the times where the location is defined
            self.timef = self.times[self.defined].value[self.inlier_mask]

            # Do the quadratic fit to the data
            try:
                self.quad_fit, self.covariance = np.polyfit(self.timef, self.locf, 2, w=1.0/(self.errorf ** 2), cov=True)
                self.fitted = True

                # Calculate the best fit line
                self.best_fit = np.polyval(self.quad_fit, self.timef)

                # Convert to km/s
                self.velocity = self.quad_fit[1] * solar_circumference_per_degree_in_km / u.s
                self.velocity_error = np.sqrt(self.covariance[1, 1]) * solar_circumference_per_degree_in_km / u.s

                # Convert to km/s/s
                self.acceleration = 2 * self.quad_fit[0] * solar_circumference_per_degree_in_km / u.s / u.s
                self.acceleration_error = 2 * np.sqrt(self.covariance[0, 0]) * solar_circumference_per_degree_in_km / u.s / u.s

                # Calculate the Long et al (2014) score
                self.long_score = aware_utils.score_long(arc.nlat, self.defined, self.velocity, self.acceleration, self.errorf, self.locf, arc.nt)

                # Fractional duration of the arc
                self.arc_duration_fraction = aware_utils.arc_duration_fraction(self.defined, arc.nt)

                # Reduced chi-squared.
                self.rchi2 = (1.0 / (1.0 * (len(self.timef) - 3.0))) * np.sum(((self.best_fit - self.locf) / self.errorf) ** 2)

                #plt.errorbar(self.timef, self.locf, yerr=self.errorf)
                #plt.plot(self.timef, self.best_fit)
                #plt.show()

            except (LA.LinAlgError, ValueError):
                # Error in the fitting algorithm
                self.fitted = False
