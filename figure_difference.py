#
# Make a figure that shows the differences between running difference,
# percentage base difference, running difference persistence images.
#

import os
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from sunpy.map import Map
import mapcube_tools
import aware
import aware_utils
import demonstration_info

plt.ion()

example = 'previous1'

# Load in all the special information needed
info = demonstration_info.info

# Where the data is
root = os.path.expanduser('~/Data/eitwave')

# Image files
imgloc = os.path.join(root, 'fts', example)

# Get the file list
lindex_lo = info[example][0]
lindex_hi = info[example][1]
l = aware_utils.get_file_list(imgloc, 'fts')[lindex_lo: lindex_hi]

# Running difference'
accum = info[example]["accum"]
mc = Map(aware_utils.accumulate_from_file_list(l, accum=1), cube=True)

rd = mapcube_tools.running_difference(mc)
data = rd[0].data
data[data < 0] = -np.sqrt(-data[data < 0])
data[data > 0] = np.sqrt(data[data > 0])

newrd = Map(data, rd[0].meta)
