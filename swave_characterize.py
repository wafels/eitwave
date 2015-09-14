#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import copy

# Main AWARE processing and detection code
import aware

# AWARE utilities
import aware_utils

# Wave simulation code
import test_wave2d

# Simulated wave parameters
import swave_params

# Mapcube handling tools
import mapcube_tools

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

# Select the wave
example = 'wavenorm4'

# What type of output do we want to analyze
mctype = 'finalmaps'

# Number of trials
ntrials = 100

# Number of images
max_steps = 80

# Accumulation in the time direction
accum = 2

# Summing in the spatial directions
spatial_summing = [4, 4]*u.pix

# Radii of the morphological operations
radii = [[5, 5], [11, 11], [22, 22]]

# Position measuring choices
position_choice = 'maximum'
error_choice = 'maxwidth'

# Unraveling factor
unraveling_factor = 1.0

# Output directory
output = '~/eitwave/'

# Output types
otypes = ['img', 'pkl']

# Special designation: an extra description added to the file and directory
# names in order to differentiate between experiments on the same example wave.
special_designation = '_test_3_only'

# Output directories and filename
odir = os.path.expanduser(output)
otypes_dir = {}
otypes_filename = {}

# Morphological radii
sradii = ''
for r in radii:
    for v in r:
        sradii = sradii + str(v) + '_'
sradii = sradii[0: -1]

# Create the storage directories and filenames
for ot in otypes:
    # root directory
    idir = os.path.join(odir, ot)

    # filename
    filename = ''

    # All the subdirectories
    for loc in [example + special_designation,
                mctype,
                str(ntrials) + '_' + str(max_steps) + '_' + str(accum) + '_' + str(spatial_summing),
                sradii,
                position_choice + '_' + error_choice]:
        idir = os.path.join(idir, loc)
        filename = filename + loc + '.'
    filename = filename[0: -1]
    if not(os.path.exists(idir)):
        os.makedirs(idir)
    otypes_dir[ot] = idir
    otypes_filename[ot] = filename

# Load in the wave params
params = swave_params.waves()[example]

# Unraveling params are different compared to the wave definition params
params_unravel = copy.deepcopy(params)
# Sum over many of the original bins used to create the wave in an attempt to
# beat down transform artifacts
params_unravel['lon_bin'] = unraveling_factor * params['lon_bin']
# Move zero location of longitudinal reconstruction relative to the
# wavefront
# params_unravel['lon_min'] = params_unravel['lon_min']
# params_unravel['lon_max'] = params_unravel['lon_max']

# Storage for the results
results = []

# Go through all the test waves, and apply AWARE.
for i in range(0, ntrials):
    # Let the user which trial is happening
    print('\nSimulating %s ' % example)
    print(' - position choice = %s' % position_choice)
    print(' - error choice = %s' % error_choice)
    print(' - unraveling factor = %f' % unraveling_factor)
    print(' - starting trial %i out of %i\n' % (i + 1, ntrials))

    # Simulate the wave and return a dictionary
    out = test_wave2d.simulate_wave2d(params=params, max_steps=max_steps,
                                      verbose=True, output=['finalmaps'])

    # Get the final map out
    mc = out['finalmaps']

    # Time when we think that the event started
    originating_event_time = mc[0].date

    # Accumulate the data in space and time to increase the signal to noise
    # ratio
    mc = mapcube_tools.accumulate(mapcube_tools.superpixel(mc, spatial_summing),
                                  accum)

    # Unravel the data
    unraveled = aware_utils.map_unravel(mc, params_unravel)

    # AWARE image processing
    umc = aware.processing(unraveled, radii=radii)

    # Get and store the dynamics of the wave front
    # Note that the error in the position of the wavefront (when measured as
    # the maximum should also include the fact that the wavefront maximum can
    # be anywhere inside that pixel
    results.append(aware.dynamics(umc,
                                  params_unravel,
                                  originating_event_time=originating_event_time,
                                  error_choice=error_choice,
                                  position_choice=position_choice,
                                  returned=['answer']))
#
# Save the results
#
if not os.path.exists(otypes_dir['pkl']):
    os.makedirs(otypes_dir['pkl'])
filepath = os.path.join(otypes_dir['pkl'], otypes_filename['pkl'] + '.pkl')
print('Results saved to %s' % filepath)
f = open(filepath, 'wb')
pickle.dump(results, f)
f.close()

