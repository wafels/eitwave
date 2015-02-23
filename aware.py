#
# Demonstration AWARE algorithm
#
import os
import numpy as np
import numpy.linalg as LA
from skimage.morphology import closing, disk
from skimage.filter.rank import median
import matplotlib.pyplot as plt
from sunpy.map import Map
from sunpy.time import parse_time
import aware_utils
import aware_plot
import mapcube_tools

# The factor below is the circumference of the sun in meters kilometers divided
# by 360 degrees.
solar_circumference_per_degree_in_km = aware_utils.solar_circumference_per_degree / 1000.0


def dump_images(mc, dir, name):
    for im, m in enumerate(mc):
        fname = '%s_%05d.png' % (name, im)
        dump_image(m.data, dir, fname)


def dump_image(img, dir, name):
    ndir = os.path.expanduser('~/eitwave/img/%s/' % dir)
    if not(os.path.exists(ndir)):
        os.makedirs(ndir)
    plt.ioff()
    plt.imshow(img)
    plt.savefig(os.path.join(ndir, name))

#
# Some potential improvements
#
# 1. Do the median filtering and the closing on multiple length-scales
#    Could add up the results at multiple length-scales to get a better idea of where the wavefront is
#
# 2. Apply the median and morphological operations on the 3 dimensional datacube, to take advantage
#    of previous and future observations.
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

            # Apply the morphological closing operation to rejoin separated parts of the wave front.
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
             returned=['arc', 'answer']):
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
    if originating_event_time == None:
        originating_event_time = unraveled[0].date
    # Displacement between the time of the originating event and the first measurement
    offset = (parse_time(unraveled[0].date) - parse_time(originating_event_time)).seconds

    # At all times get an average location of the wavefront
    nlon = data.shape[1]
    nlat = data.shape[0]
    latitude = np.min(unraveled[0].yrange) + np.arange(0, nlat) * lat_bin

    results = []
    for lon in range(0, nlon):
        arc = Arc(data[:, lon, :], times, latitude, offset)
        answer = FitPosition(arc, error_choice=error_choice, position_choice=position_choice)
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
        self.times = times + self.offset
        self.nt = times.size
        self.nlat = latitude.size
        self.lat_bin = latitude[1] - latitude[0]
        self.latitude = np.arange(0, self.nlat) * self.lat_bin
        self.title = title

    def peek(self):
        return aware_plot.arc(self)


class FitPosition:
    """
    Fit the average position of the wavefront along an arc.
    :param arc:
    :param error_choice:
    :return:
    """

    def __init__(self, arc, error_choice='std', position_choice='average'):
        # Is the arc fit-able?  Assume that it is.
        self.fitable = True

        # Has the arc been fitted?
        self.fitted = False

        # Temporal offset - difference between the originating event time and the first possible measurement of
        # the arc
        self.offset = arc.offset

        # Which error measurement of the position to use when determining the wave
        self.error_choice = error_choice

        # Which position to use when determining the wave
        self.position_choice = position_choice

        # Average position of the remaining emission at each time
        self.avpos = np.zeros([arc.nt])

        # Position of the wave maximum at each time
        self.maximum = np.zeros_like(self.avpos)

        # Standard deviation of the remaining emission at each time
        self.std = np.zeros_like(self.avpos)

        # Maximum extent of remaining emission as measured by subtracting
        # the emission closest to the start from the emission furthest from the start
        self.maxwidth = np.zeros_like(self.avpos)
        for i in range(0, arc.nt):
            emission =  arc.data[::-1, i]
            summed_emission = np.sum(emission)
            nonzero_emission = np.nonzero(emission)
            self.avpos[i] = np.sum(emission * arc.latitude) / summed_emission
            self.std[i] = np.std(emission * arc.latitude) / summed_emission
            self.maximum[i] = arc.latitude[np.argmax(emission)]
            if len(nonzero_emission[0]) > 0:
                self.maxwidth[i] = arc.lat_bin * (nonzero_emission[0][-1] - nonzero_emission[0][0])
            else:
                self.maxwidth[i] = 0.0

        # Wave sample times
        self.times = arc.times

        # Choose which characterization of the wavefront to use
        if self.position_choice == 'average':
            self.pos = self.avpos
        if self.position_choice == 'maximum':
            self.pos = self.maximum

        # Locations of the finite data
        self.pos_isfinite = np.isfinite(self.pos)

        # There is at least one location that has emission greater than zero
        self.at_least_one_nonzero_location = np.any(self.pos[self.pos_isfinite] > 0.0)

        # Error choice
        if self.error_choice == 'std':
            self.error = self.std
        if self.error_choice == 'maxwidth':
            self.error = self.maxwidth

        # Find if we have enough points to do a quadratic fit
        self.error_isfinite = np.isfinite(self.error)
        self.defined = self.pos_isfinite * self.error_isfinite * self.at_least_one_nonzero_location
        if np.sum(self.defined) <= 3:
            self.fitable = False

        if self.fitable:
            # Get the times where the location is defined
            self.timef = self.times[self.defined]
            # Get the locations where the location is defined
            self.locf = self.pos[self.defined]
            # Get the standard deviation where the location is defined
            self.errorf = self.error[self.defined]

            # Do the quadratic fit to the data
            try:
                print self.timef, self.locf, self.errorf
                self.quadfit, self.covariance = np.polyfit(self.timef, self.locf, 2, w=self.errorf, cov=True)
                self.fitted = True
                # Calculate the best fit line
                self.bestfit = np.polyval(self.quadfit, self.timef)
                # Convert to km/s
                self.velocity = self.quadfit[1] * solar_circumference_per_degree_in_km
                self.velocity_error = np.sqrt(self.covariance[1, 1]) * solar_circumference_per_degree_in_km
                # Convert to km/s/s
                self.acceleration = 2 * self.quadfit[0] * solar_circumference_per_degree_in_km
                self.acceleration_error = 2 * np.sqrt(self.covariance[0, 0]) * solar_circumference_per_degree_in_km
                # Calculate the Long et al (2014) score
                self.long_score = aware_utils.score_long(arc.nlat, self.defined, self.velocity, self.acceleration, self.errorf, self.locf, arc.nt)
                # Fractional duration of the arc
                self.arc_duration_fraction = aware_utils.arc_duration_fraction(self.defined, arc.nt)
                # Reduced chi-squared.
                self.rchi2 = (1.0 / (1.0 * (len(self.timef) - 3.0))) * np.sum(((self.bestfit - self.locf) / self.errorf) ** 2)
            except LA.LinAlgError:
                # Error in the fitting algorithm
                self.fitted = False
