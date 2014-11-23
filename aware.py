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

    # Get rid of everything below zero
    new = Map([Map(np.clip(m.data, 0.0, np.max(m.data)), m.meta) for m in new], cube=True)

    # Get the square root
    new = Map([Map(np.sqrt(m.data), m.meta) for m in new], cube=True)

    # Get rid of spikes
    new = Map([Map(np.clip(m.data, np.min(m.data), prop["spike_level"] * prop["accum"]), m.meta) for m in new], cube=True)
        
    # Get rid of noise by applying the median filter
    median_disk = disk(prop["median_radius"])
    new = Map([Map(median(m.data, median_disk), m.meta) for m in new], cube=True)

    # Apply the morphological closing operation to rejoin separated parts of the wave front.
    closing_disk = disk(prop["closing_radius"])
    new = Map([Map(closing(m.data, closing_disk), m.meta) for m in new], cube=True)

    # Return the cleaned mapcube
    return new


def dynamics():
    """
    Measurement of the progress of the wave across the disk.  This part of
    AWARE generates information concerning the dynamics of the wavefront.
    """
    pass