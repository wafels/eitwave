#
# Demonstration AWARE algorithm
#

import numpy as np
from sunpy.map import Map
from skimage.morphology import closing, disk
from skimage.filter.rank import median
import mapcube_tools

def processing(mc, prop):
    """
    Image processing steps used to isolate the EUV wave from the data.  Use
    this part of AWARE to perform the image processing steps that segment
    propagating features that brighten new pixels as they propagate.

    Parameters
    ----------

    mc : sunpy.map.MapCube
    prop : a dictionary that holds variables that control the analysis

    """
    # Calculate the persistence
    new = mapcube_tools.persistence(mc)
    
    # Calculate the running difference
    new = mapcube_tools.running_difference(new)

    # Define the
    median_disk = disk(prop["median_radius"])
    closing_disk = disk(prop["closing_radius"])

    newmc = []
    for m in new:

        # Get rid of everything below zero
        newdata = np.clip(m.data, 0.0, np.max(m.data))

        # Get the square root
        newdata = np.sqrt(newdata)

        # Get rid of spikes
        newdata = np.clip(newdata, np.min(newdata), prop["spike_level"] * prop["accum"])

        # Get rid of noise by applying the median filter
        newdata = median(newdata, median_disk)

        # Apply the morphological closing operation to rejoin separated parts of the wave front.
        new = closing(newdata, closing_disk)

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