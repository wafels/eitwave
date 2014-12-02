#
# Trying to remove a lot of extraneous structure in the detection of
# EIT / EUV waves
#
import os
from copy import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from sunpy.net import hek
from sunpy.map import Map
import aware
import demonstration_info

import aware_utils
from visualize import visualize_dc, visualize

plt.ion()

# Examples to look at
#example = 'previous1'
#example = 'corpita_fig4'
#example = 'corpita_fig6'
#example = 'corpita_fig7'
example = 'corpita_fig8a'
#example = 'corpita_fig8e'

# Load in all the special information needed
info = demonstration_info.info

# Where the data is
root = os.path.expanduser('~/Data/eitwave')

# Image files
imgloc = os.path.join(root, 'fts', example)



#
# Full AWARE algorithm would start with identifying an event, downloading the dat
# and then running the identification and dynamics code
#


# Get the file list
l = aware_utils.get_file_list(imgloc, 'fts')

# Increase signal to noise ratio
print example + ': Accumulating images'
accum = info[example]["accum"]
mc = Map(aware_utils.accumulate_from_file_list(l, accum=accum), cube=True)

# Image processing
transformed = aware.processing(mc)

# HEK flare results
print('Getting HEK flare results.')
pickleloc = os.path.join(root, 'pkl', example)
hekflarename = example + '.hek.pkl'
pkl_file_location = os.path.join(pickleloc, hekflarename)
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


# Get the location of the source event
params = aware_utils.params(result[info[example]['result']])

# Unravel the data.  Note that the first element of the transformed array is, in these examples at least, not a good
# representation of the wavefront.  It is there removed when calculating the unraveled maps
umc = aware.unravel(transformed[1:], params)

# Get the dynamics of the wave front
dynamics = aware.dynamics(umc, params)

#
# Recover the scores
#
long_scores = []
for i, r in enumerate(dynamics):
    if r != None:
        long_scores.append((i, r["long_score"]))

#
# Summary stats of the Long et al scores - these measure the wave quality.
#
all_long_scores = np.asarray([score[1] for score in long_scores])
long_score_arithmetic_mean = np.mean(all_long_scores)
long_score_geometric_mean = np.exp(np.mean(np.log(all_long_scores)))


#
# Find where the best scores are
#
max_score = np.max(all_long_scores)
best_lon = []
for i, r in enumerate(long_scores):
        if r[1] == max_score:
            best_lon.append(r[0])

#
# Having identified where the best longitudes are, we can plot out curves of
# of the progress of the wave front
#
