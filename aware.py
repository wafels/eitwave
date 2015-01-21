#
# Demonstration AWARE algorithm
#
import pickle
import numpy as np
import numpy.linalg as LA
from skimage.morphology import closing, disk
from skimage.filter.rank import median
import matplotlib.pyplot as plt
from sunpy.map import Map
from sunpy.time import parse_time
import aware_utils
import mapcube_tools

# The factor below is the circumference of the sun in meters kilometers divided
# by 360 degrees.
solar_circumference_per_degree = 1.21e4

def dump_images(mc, name):
    for im, m in enumerate(mc):
        fname = '%s_%05d.png' % (name, im)
        dump_image(m.data, fname)


def dump_image(img, name):
    plt.ioff()
    plt.imshow(img)
    plt.savefig('%s.png' % name)

#
# Some potential improvements
#
# 1. Do the median filtering and the closing on multiple length-scales
#    Could add up the results at multiple length-scales to get a better idea of where the wavefront is
#
# 2. Apply the median and morphological operations on the 3 dimensional datacube, to take advantage
#    of previous and future observations.
#
def processing(mc, median_radius=11, closing_radius=11, spike_level=25, accum=1):
    """
    Image processing steps used to isolate the EUV wave from the data.  Use
    this part of AWARE to perform the image processing steps that segment
    propagating features that brighten new pixels as they propagate.

    Parameters
    ----------

    mc : sunpy.map.MapCube
    median_radius :
    closing_radius :
    spike_level :
    accum :
    """
    # Calculate the persistence
    new = mapcube_tools.persistence(mc)
    dump_images(new, '1_persistence')

    # Calculate the running difference
    new = mapcube_tools.running_difference(new)
    dump_images(new, '2_rdiff')

    # Define the
    median_disk = disk(median_radius)
    closing_disk = disk(closing_radius)

    newmc = []
    for im, m in enumerate(new):

        # Get rid of everything below zero
        newdata = np.clip(m.data, 0.0, np.max(m.data))
        dump_image(newdata, '3_clip_%05d' % im)

        # Get the square root
        newdata = np.sqrt(newdata)
        dump_image(newdata, '4_sqrt_%05d' % im)

        # Get rid of spikes
        newdata = np.clip(newdata, np.min(newdata), spike_level * accum)
        dump_image(newdata, '5_clip_%05d' % im)

        # Get rid of noise by applying the median filter.  This implementation of the median filter
        # requires that the data be scaled between 0 and 1.
        newdata = newdata / np.max(newdata)
        newdata = median(newdata, median_disk)
        dump_image(newdata, '6_median_%05d' % im)

        # Apply the morphological closing operation to rejoin separated parts of the wave front.
        newdata = closing(newdata, closing_disk)
        dump_image(newdata, '7_closing_%05d' % im)

        # New mapcube list
        newmc.append(Map(newdata, m.meta))



    # Return the cleaned mapcube
    return Map(newmc, cube=True)


def _get_times_from_start(mc):
    # Get the times of the images
    start_time = parse_time(mc[0].date)
    return np.asarray([(parse_time(m.date) - start_time).seconds for m in mc])


def unravel(mc, params):
    """
    
    """
    return aware_utils.map_unravel(mc, params)


def dynamics(unraveled, params, error_choice='std'):
    """
    Measurement of the progress of the wave across the disk.  This part of
    AWARE generates information concerning the dynamics of the wavefront.
    """
    # Size of the latitudinal bin
    lat_bin = params.get('lat_bin')
    # Get the data
    data = unraveled.as_array()
    # Times
    times = _get_times_from_start(unraveled)

    # At all times get an average location of the wavefront
    nlon = data.shape[1]
    nlat = data.shape[0]
    latitude = np.min(unraveled[0].yrange) + np.arange(0, nlat) * lat_bin

    results = []
    for lon in range(0, nlon):
        answer = FitAveragePosition(Arc(data[:, lon, :], times, latitude), error_choice=error_choice)
        # Store the collated results
        results.append(answer)

    return results


class Arc:
    """
    Object to store data on the emission along each arc as a function of time
    data : ndarray of size (nlat, nt)
    times : ndarray of time in seconds from zero
    latitude : ndarray of the latitude bins of the unraveled array
    """
    def __init__(self, data, times, latitude, title=''):
        self.data = data
        self.times = times
        self.latitude = latitude
        self.title = title
        self.nlat = latitude.size
        self.lat_bin = self.latitude[1] - self.latitude[0]
        self.nt = times.size

    def peek(self):
        plt.imshow(self.data)
        plt.ylabel('latitude index')
        plt.xlabel('time index')
        plt.title('arc' + self.title)
        plt.show()


class FitAveragePosition:
    """
    Fit the average position of the wavefront along an arc.
    :param arc:
    :param error_choice:
    :return:
    """


    def __init__(self, arc, error_choice='std', use_increasing=False, use_increasing_factor=10000.0):
        self.attempted_fit = True
        # Which error measurement of the position to use when determining the wave
        self.error_choice = error_choice
        #
        self.use_increasing = use_increasing
        #
        self.use_increasing_factor = use_increasing_factor
        # Average position of the remaining emission at each time
        self.avpos = np.zeros([arc.nt])
        # Standard deviation of the remaining emission at each time
        self.std = np.zeros_like(self.avpos)
        # Maximum extent of remaining emission as measured by subtracting
        # the emission closest to the start from the emission furthest from the start
        self.maxwidth = np.zeros_like(self.avpos)
        for i in range(0, arc.nt):
            emission = np.sum(arc.data[:, i])
            summed_emission = np.sum(emission)
            nonzero_emission = emission != 0.0
            self.avpos[i] = np.sum(emission * arc.latitude) / summed_emission
            self.std[i] = np.std(emission * arc.latitude) / summed_emission
            self.maxwidth[i] = arc.lat_bin * np.max(np.asarray([1.0, np.argmax(nonzero_emission) - np.argmin(nonzero_emission)]))

        # Locations of the finite data
        self.avpos_isfinite = np.isfinite(self.avpos)
        # There is at least one location that has emission greater than zero
        self.at_least_one_nonzero_location = np.any(self.avpos[self.avpos_isfinite] > 0.0)

        if self.error_choice == 'std':
            self.error = self.std
        if self.error_choice == 'extent':
            self.error = self.maxwidth

        # Find if we have enough points to do a quadratic fit
        self.error_isfinite = np.isfinite(self.error)
        self.defined = self.avpos_isfinite * self.error_isfinite * self.at_least_one_nonzero_location
        if np.sum(self.defined) <= 3:
            return None

        # Get the times where the location is defined
        self.timef = arc.times[self.defined]
        # Get the locations relative to the first position where the location is defined
        self.locf = np.abs(self.avpos[self.defined] - self.avpos[self.defined][0])
        # Get the standard deviation where the location is defined
        self.errorf = self.error[self.defined]
        # Get where the data is increasing relative to the previous one
        self.increasing = np.zeros_like(self.defined)
        self.increasing[1:] = np.asarray(np.diff(self.locf) >= 0.0)[:]
        # What to do with data that does not increase relative to the previous one.
        if self.use_increasing:
            self.errorf[not(self.increasing)] = self.use_increasing_factor * self.errorf[not(self.use_increasing)]

        # Do the quadratic fit to the data
        try:
            self.quadfit, self.covariance = np.polyfit(self.timef, self.locf, 2, w=self.errorf, cov=True)
            self.fitted = True
            # Calculate the best fit line
            self.bestfit = np.polyval(self.quadfit, self.timef)
            # Convert to km/s
            self.velocity = self.quadfit[1] * solar_circumference_per_degree
            # Convert to km/s/s
            self.acceleration = 2 * self.quadfit[0] * solar_circumference_per_degree
            # Calculate the Long et al (2014) score
            self.long_score = aware_utils.score_long(arc.nlat, self.defined, self.velocity, self.acceleration, self.errorf, self.locf, self.nt)
            #
            self.arc_duration_fraction = aware_utils.arc_duration_fraction(self.defined, arc.nt)
        except LA.LinAlgError:
            # Error in the fitting algorithm
            self.fitted = False

