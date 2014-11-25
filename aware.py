#
# Demonstration AWARE algorithm
#

import numpy as np
from sunpy.map import Map
from skimage.morphology import closing, disk
from skimage.filter.rank import median
import mapcube_tools
import matplotlib.pyplot as plt

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


def dynamics():
    """
    Measurement of the progress of the wave across the disk.  This part of
    AWARE generates information concerning the dynamics of the wavefront.
    """
    pass


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