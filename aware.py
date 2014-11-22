#
# Demonstration AWARE algorithm
#

import numpy as np
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
    a = mapcube_tools.persistence(mc)
    
    # Calculate the running difference
    a = mapcube_tools.running_difference(a)

    # Get rid of everything below zero
    a = mapcube_tools.apply_to_each_map(a, np.clip, 0.0, np.max(a))

    # Get the square root
    a = mapcube_tools.apply_to_each_map(a, np.sqrt)

    # Get rid of spikes
    a = mapcube_tools.apply_to_each_map(a, np.clip, np.min(a), 25 * prop["accum"])
        
    # Get rid of noise by applying the median filter
    
    # Apply the morphological closing operation to rejoin separated parts of
    # the wave front.
    
    # Return the cleaned macpcube
    return b


def dynamics():
    """
    Measurement of the progress of the wave across the disk.  This part of
    AWARE generates information concerning the dynamics of the wavefront.
    """
    pass