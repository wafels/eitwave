#
# Trying to remove a lot of extraneous structure in the detection of
# EIT / EUV waves
#
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from sunpy.net import hek
from sunpy.map import Map
import aware
import demonstration_info
import test_wave2d

import aware_utils
from visualize import visualize_dc, visualize

plt.ion()

# Examples to look at
#example = 'previous1'
example = 'simulatelownoise'
#example = 'corpita_fig4'
#example = 'corpita_fig6'
#example = 'corpita_fig7'
#example = 'corpita_fig8a'
#example = 'corpita_fig8e'

# Load in all the special information needed
info = demonstration_info.info

# Where the data is
root = os.path.expanduser('~/Data/eitwave')

# FITS files location
imgloc = os.path.join(root, 'fts', example)

# Pickle file output
pickleloc =os.path.join(root, 'pkl', example)
if not os.path.exists(pickleloc):
    os.makedirs(pickleloc)

#
# Full AWARE algorithm would start with identifying an event, downloading the dat
# and then running the identification and dynamics code
#
# Get the file list
l = aware_utils.get_file_list(imgloc, 'fts')
if len(l) == 0:
    l = aware_utils.get_file_list(imgloc, 'fits')

# Increase signal to noise ratio
print example + ': Accumulating images'
accum = info[example]["accum"]
mc = Map(aware_utils.accumulate_from_file_list(l, accum=accum, nsuper=1,verbose=True), cube=True)

# Image processing
transformed = aware.processing(mc, radii=[[11, 11]])

if example != 'simulate' and example != 'simulatelownoise':
    # HEK flare results
    print('Getting HEK flare results.')
    hekflarename = example + '.hek.pkl'
    pkl_file_location = os.path.join(pickleloc, hekflarename)
    if not os.path.isfile(pkl_file_location):
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
else:
    test_wave2d_params = test_wave2d.params
    result = [{"event_coordunit": "degrees",
               "event_coord1": test_wave2d_params['epi_lon'],
               "event_coord2": test_wave2d_params['epi_lat']}]

# Get the location of the source event
params = aware_utils.params(result[info[example]['result']])

# Unravel the data.  Note that the first element of the transformed array is, in these examples at least, not a good
# representation of the wavefront.  It is there removed when calculating the unraveled maps
umc = aware.unravel(transformed[1:], params)

f = open(os.path.join(pickleloc, 'umc_%s.pkl' % example), 'wb')
pickle.dump(umc, f)
f.close()


# Get the dynamics of the wave front
dynamics = aware.dynamics(umc, params)

#
# Recover the scores
#
score_key = "rchi2"
assessment_scores = []
for i, r in enumerate(dynamics):
    if r[1].fitted:
        assessment_scores.append((i, r[1].rchi2))
    else:
        assessment_scores.append((i, 0.0))

#
# Summary stats of the Long et al scores - these measure the wave quality.
#
all_assessment_scores = np.asarray([score[1] for score in assessment_scores])
assessment_score_arithmetic_mean = np.mean(all_assessment_scores)
assessment_score_geometric_mean = np.exp(np.mean(np.log(all_assessment_scores)))


#
# Find where the best scores are
#
max_score = np.max(all_assessment_scores)
best_lon = []
for i, r in enumerate(assessment_scores):
        if r[1] == max_score:
            best_lon.append(r[0])

#
# Go through all the arcs and find the biggest continuous range of detections.
# This can be used to determine if a wave has been detected, or to what level
# we are sure that a wave has been detected
#



#
# Having identified where the best longitudes are, we can plot out curves of
# of the progress of the wave front
#
for ibest, best in enumerate(best_lon):
    t = dynamics[best][1].timef#["timef"]
    error = dynamics[best][1].errorf#["stdf"]
    locf = dynamics[best][1].locf#["locf"]
    bestfit = dynamics[best][1].bestfit#["bestfit"]
    bestlong = best #dynamics[best]["longitude"]
    plt.figure(ibest)
    # plot the location of the wave and its error
    plt.errorbar(t, locf, yerr=(error, error),
                 fmt='ro',
                 label='wave location')
    # plot the fit to the data
    plt.plot(t, bestfit, label='best fit')
    # Label the plot
    plt.xlabel('time (seconds)')
    plt.ylabel('degrees of arc from wave origin')
    plt.title(example + ': arc # %i' % (bestlong))
    # Display the measured velocity and acceleration with errors
    
    # Display the Long et al. score for this arc
    
    # Make the legend
    plt.legend(loc=4)
    # Save the data out.
    plt.savefig(example + '.dynamics.' + str(bestlong) + '.png')
