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


def do_fit_to_data(nlat, times, thisloc, std):
    """
    Do a quadratic fit to the data
    :param thisloc:
    :param std:
    :return:

    """
    # Number of sample times
    nt = len(times)

    # Find where the location is defined
    loc_exists = np.isfinite(thisloc)
    if len(loc_exists) == 0:
        return None

    # Find if we have enough points to do a quadratic fit
    defined = loc_exists * np.isfinite(std) * np.any(thisloc[loc_exists] > 0.0)
    if np.sum(defined) <= 3:
        return None

    # Get the times where the location is defined
    timef = times[defined]
    # Get the locations where the location is defined
    locf = thisloc[defined]
    # Get the locations relative to the first position
    locf = np.abs(locf - locf[0])
    # Get the standard deviation where the location is defined
    stdf = std[defined]

    # Do the quadratic fit to the data
    try:
        quadfit, covariance = np.polyfit(timef, locf, 2, w=stdf, cov=True)
        # Calculate the best fit line
        bestfit = np.polyval(quadfit, timef)
        # Convert to km/s
        velocity = quadfit[1] * solar_circumference_per_degree
        # Convert to km/s/s
        acceleration = 2 * quadfit[0] * solar_circumference_per_degree
        # Calculate the Long et al (2014) score
        long_score = aware_utils.score_long(nlat, defined, velocity, acceleration, stdf, locf, nt)
        # 
        arc_duration_fraction = aware_utils.arc_duration_fraction(defined, nt)
        # create a dictionary that stores the results and append it
        return {"bestfit": bestfit, "quadfit": quadfit, "covariance": covariance,
                "velocity": velocity, "acceleration": acceleration,
                "stdf": stdf, "locf": locf, "timef": timef,
                "long_score": long_score,
                "arc_duration_fraction": arc_duration_fraction}
    except LA.LinAlgError:
        # Error in the fitting algorithm
        return None


def dynamics(unraveled, params):
    """
    Measurement of the progress of the wave across the disk.  This part of
    AWARE generates information concerning the dynamics of the wavefront.
    """
    # Get the times of the images
    start_time = parse_time(unraveled[0].date)
    times = np.asarray([(parse_time(m.date) - start_time).seconds for m in unraveled])

    # Get the data
    data = unraveled.as_array()

    # At all times get an average location of the wavefront
    nlon = data.shape[1]
    nlat = data.shape[0]
    nt = len(times)
    latitude = np.min(unraveled[0].yrange) + np.arange(0, nlat) * params.get('lat_bin')

    results = []
    for lon in range(0, nlon):
        thisloc = np.zeros([nt])
        std = np.zeros_like(thisloc)
        for i in range(0, nt):
            emission = data[:, lon, i]
            summed_emission = np.sum(emission)
            # Simple estimate of where the bulk of the wavefront is
            thisloc[i] = np.sum(emission * latitude) / summed_emission
            std[i] = np.std(emission * latitude) / summed_emission

        # Fit a quadratic to the position estimate
        answer = do_fit_to_data(nlat, times, thisloc, std)
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