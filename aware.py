#
# Demonstration AWARE algorithm
#

import numpy as np

def processing(mc):
    """
    Image processing steps used to isolate the EUV wave from the data
    """
    # Calculate the persistence
    a = mapcube_tools.persistence(mc)
    
    # Calculate the running difference
    a = mapcube_tools.running_difference(a)

    # Get rid of everything below zero
    a = mapcube_tools.simple_threshold(a, 'lt', 0.0, 0.0)

    # Get the square root
    a = mapcube_tools.apply(a, np.sqrt)

    # Get rid of spikes
    a = mapcube_tools.simple_threshold(a, 'gt', 25 * accum, 25 * accum)
        
    # Get rid of noise by applying the median filter
    
    # Apply the morphological closing operation to rejoin separated parts of
    # the wave front.
    
    # Return the cleaned macpcube
    return b


def dynamics(???):
    """
    Measurement of the progress of the wave across the disk.
    """
    pass