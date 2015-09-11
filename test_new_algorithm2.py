#
# Trying to remove a lot of extraneous structure in the detection of
# EIT / EUV waves
#
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from sunpy.net import hek
from sunpy.map import Map

import aware
import demonstration_info
import test_wave2d
import aware_utils
import aware_plot
import swave_params

plt.ion()

# Simulated data
# TODO - run the same analysis on multiple noisy realizations of the simulated
# TODO - data. Generate average result plots over the multiple realizations
# TODO - either as relative error or absolute error as appropriate.
# TODO - First mode: use multiple noisy realizations of the same model.
# TODO - Second mode: choose a bunch of simulated data parameters, and then
# TODO - randomly select values for them (within reason).
# TODO - The recovered parameters should be reasonably close to the simulated
# TODO - parameters.
#
#

simulated = ['sim_speed', 'sim_half_speed', 'sim_double_speed', 'sim_speed_and_dec', 'sim_speed_and_acc']

# Position measuring choices
position_choice = 'maximum'
error_choice = 'maxwidth'

# Image output directory
imgdir = os.path.expanduser('~/eitwave/img/')
if not(os.path.exists(imgdir)):
    os.makedirs(imgdir)

# Examples to look at
#example = 'previous1'
example = simulated[0]
#example = 'corpita_fig4'
#example = 'corpita_fig6'
#example = 'corpita_fig7'
#example = 'corpita_fig8a'
#example = 'corpita_fig8e'

# Load in all the special information needed
if example in simulated:
    info = {"tr": hek.attrs.Time('2021-06-07 06:15:00', '2021-06-07 07:00:00'),
            "accum": 2,
            "result": 0}
    simulated_params = {"true_velocity": 933.0, "true_acceleration": 0.0}
else:
    info = demonstration_info.info[example]
    simulated_params = None


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
accum = info["accum"]
originating_event_time = Map(l[0]).date
mc = Map(aware_utils.accumulate_from_file_list(l,
                                               accum=accum,
                                               nsuper=[1,1]*u.pix,
                                               verbose=True),
         cube=True)

# Get the originating location
if not(example in simulated):
    # HEK flare results
    print('Getting HEK flare results.')
    hekflarename = example + '.hek.pkl'
    pkl_file_location = os.path.join(pickleloc, hekflarename)
    if not os.path.isfile(pkl_file_location):
        hclient = hek.HEKClient()
        tr = info["tr"]
        ev = hek.attrs.EventType('FL')
        oresult = hclient.query(tr, ev, hek.attrs.FRM.Name == 'SSW Latest Events')
        pkl_file = open(pkl_file_location, 'wb')
        pickle.dump(oresult, pkl_file)
        pkl_file.close()
    else:
        pkl_file = open(pkl_file_location, 'rb')
        oresult = pickle.load(pkl_file)
        pkl_file.close()
else:
    test_wave2d_params = swave_params.waves()['basic_wave']
    oresult = [{"event_coordunit": "degrees",
               "event_coord1": test_wave2d_params['epi_lon'],
               "event_coord2": test_wave2d_params['epi_lat']}]

# Get the location of the source event
params = aware_utils.params(oresult[info['result']])


# Unravel the data
unraveled = aware.unravel(mc, params)

# Unravel the data.  Note that the first element of the transformed array is, in these examples at least, not a good
# representation of the wavefront.  It is there removed when calculating the unraveled maps

umc = aware.processing(unraveled, radii=[[11, 11], [5, 5], [22, 22]])

f = open(os.path.join(pickleloc, 'umc_%s.pkl' % example), 'wb')
pickle.dump(umc, f)
f.close()

# Get the dynamics of the wave front
dynamics = aware.dynamics(Map(umc[1:], cube=True), params,
                          originating_event_time=originating_event_time,
                          error_choice=error_choice, position_choice=position_choice)

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
# Plot out summary dynamics for all the arcs
#
aware_plot.all_arcs_summary_plots(imgdir, example, dynamics, simulated_params=simulated_params)

