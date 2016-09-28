#
# Script that loads in a data set and creates a series of plots
# that compare the effect of different running difference algorithms

import numpy as np
import mapcube_tools

for location in locations:


    # running difference
    mc_rd = mapcube_tools.running_difference(mc)

    # fraction base difference
    mc_pbd = mapcube_tools.base_difference(mc, fraction=True)

    # running difference persistence images
    mc_rdpi = mapcube_tools.running_difference(mapcube_tools.persistence(mc))

