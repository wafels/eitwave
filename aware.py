#
# Demonstration AWARE algorithm
#

import numpy as np
import numpy.linalg as LA
from sunpy.map import Map
from sunpy.time import parse_time
from skimage.morphology import closing, disk
from skimage.filter.rank import median
import mapcube_tools
import matplotlib.pyplot as plt
import aware_utils

# The factor below is the circumference of the sun in meters kilometers divided
# by 360 degrees.
solar_circumference_per_degree = 1.21e4


def get_trigger_events(eventname):
    """
    Function to obtain potential wave triggering events from the HEK.
    """
    # Main directory holding the results
    pickleloc = aware_utils.storage(eventname)
    # The filename that stores the triggering event
    hek_trigger_filename = aware_utils.storage(eventname, hek=True)
    pkl_file_location = os.path.join(pickleloc, hek_trigger_name)
    
    if not os.path.exists(pickleloc):
        os.makedirs(pickleloc)
        hclient = hek.HEKClient()
        tr = info[example]["tr"]
        ev = hek.attrs.EventType('FL')
        result = hclient.query(tr, ev, hek.attrs.FRM.Name == 'SSW Latest Events')
        pkl_file = open(pkl_file_location, 'wb')
        pickle.dump(result, pkl_file)
        pkl_file.close()
    else:
        pkl_file = open(pkl_file_location, 'rb')
        result = pickle.load(pkl_file)
        pkl_file.close()


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
    
    # Calculate the running difference
    new = mapcube_tools.running_difference(new)

    # Define the
    median_disk = disk(median_radius)
    closing_disk = disk(closing_radius)

    newmc = []
    for m in new:

        # Get rid of everything below zero
        newdata = np.clip(m.data, 0.0, np.max(m.data))

        # Get the square root
        newdata = np.sqrt(newdata)

        # Get rid of spikes
        newdata = np.clip(newdata, np.min(newdata), spike_level * accum)

        # Get rid of noise by applying the median filter.  This implementation of the median filter
        # requires that the data be scaled between 0 and 1.
        newdata = newdata / np.max(newdata)
        newdata = median(newdata, median_disk)

        # Apply the morphological closing operation to rejoin separated parts of the wave front.
        newdata = closing(newdata, closing_disk)

        # New mapcube list
        newmc.append(Map(newdata, m.meta))

    # Return the cleaned mapcube
    return Map(newmc, cube=True)


def unravel(mc, params):
    """
    
    """
    return aware_utils.map_unravel(mc, params)


def dynamics(unraveled, params):
    """
    Assess the dynamics of the wavefront

    :param unraveled:
    :param params:
    :return:
    """
    # Measure the location of the wavefont
    return get_wave_front(unraveled, params)


def _get_times_from_start(mc):
    # Get the times of the images
    start_time = parse_time(mc[0].date)
    times = np.asarray([(parse_time(m.date) - start_time).seconds for m in mc])


def get_wave_front(unraveled, params, error_choice='std'):
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
    nt = data.shape[2]
    latitude = np.min(unraveled[0].yrange) + np.arange(0, nlat) * lat_bin

    results = []
    for lon in range(0, nlon):
        answer = FitAveragePosition(Arc(data[:, lon, :], times, latitude), error_choice=error_choice)
        # Store the collated results
        results.append(answer)

    return results


def write_movie(mc, filename, start=0, end=None):
    """
    Write a movie standard movie out from the input datacube

    Parameters
    ----------
    :param mc: input mapcube
    :param filename: name of the movie file
    :param start: first element in the mapcube
    :param end: last element in the mapcube
    :return: output_filename: filename of the movie.
    """
    FFMpegWriter = animation.writers['ffmpeg']
    fig = plt.figure()
    metadata = dict(title=name, artist='Matplotlib', comment='AWARE')
    writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=2000.0)
    output_filename = filename + '.mp4'
    with writer.saving(fig, output_filename, 100):
        for i in range(start, len(mc)):
            mc[i].plot()
            mc[i].draw_limb()
            mc[i].draw_grid()
            plt.title(mc[i].date)
            writer.grab_frame()
    return output_filename


class Arc:
    """
    Object to store data on the emission along each arc as a function of time
    data : ndarray of size (nlat, nt)
    times : ndarray of time in seconds from zero
    latitude : ndarray of the latitude bins of the unraveled array
    """
    def __init__(self, data, times, latitude):
        self.data = data
        self.times = times
        self.latitude = latitude
        self.nlat = latitude.size
        self.lat_bin = self.latitude[1] - self.latitude[0]
        self.nt = times.size


class FitAveragePosition:
    def __init__(self, arc, error_choice='std'):
        """
        Fit the average position of the wavefront along an arc.
        :param arc:
        :param error_choice:
        :return:
        """
        # Which error measurement of the position to use when determining the wave
        self.error_choice = error_choice
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
            self.avpos[i] = np.sum(emission * arc.latitude) / summed_emission
            self.std[i] = np.std(emission * arc.latitude) / summed_emission
            self.maxwidth[i] = arc.lat_bin * np.max(np.asarray([1.0, np.argmax(emission) - np.argmin(emission)]))

        # Locations of the finite data
        self.avpos_isfinite = np.isfinite(self.avpos)
        # There is at least one location that has emission greater than zero
        self.at_least_one_nonzero_location = np.any(self.avpos[self.avpos_isfinite] > 0.0)

        if self.error_choice == 'std':
            self.error = arc.std
        if self.error_choice == 'extent':
            self.error = arc.maxwidth

        # Find if we have enough points to do a quadratic fit
        self.error_isfinite = np.isfinite(self.error)
        self.defined = arc.avpos_isfinite * self.error_isfinite * arc.at_least_one_nonzero_location
        if np.sum(self.defined) <= 3:
            return None

        # Get the times where the location is defined
        self.timef = arc.times[self.defined]
        # Get the locations where the location is defined
        self.locf = self.avpos[self.defined]
        # Get the locations relative to the first position
        self.locf = np.abs(self.locf - self.locf[0])
        # Get the standard deviation where the location is defined
        self.errorf = self.error[self.defined]

        # Do the quadratic fit to the data
        try:
            self.quadfit, covariance = np.polyfit(self.timef, self.locf, 2, w=self.errorf, cov=True)
            self.fitted = True
            # Calculate the best fit line
            self.bestfit = np.polyval(self.quadfit, self.timef)
            # Convert to km/s
            self.velocity = self.quadfit[1] * solar_circumference_per_degree
            # Convert to km/s/s
            self.acceleration = 2 * self.quadfit[0] * solar_circumference_per_degree
            # Calculate the Long et al (2014) score
            self.long_score = aware_utils.score_long(arc.nlat, self.defined, self.velocity, self.acceleration, self.errorf, self.locf, arc.nt)
            #
            self.arc_duration_fraction = aware_utils.arc_duration_fraction(self.defined, arc.nt)
        except LA.LinAlgError:
            # Error in the fitting algorithm
            self.fitted = False

class FitUsingGaussian:
    def __init__(self, arc):
        """
        Fit a Gaussian to the wavefront
        :param arc:
        :return:
        """
        pass