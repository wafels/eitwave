#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import matplotlib.pyplot as plt

from sunpy.map import Map

# Main AWARE processing and detection code
import aware

# AWARE utilities
import aware_utils

# Wave simulation code
import test_wave2d

# Plotting code for AWARE
import aware_plot

# Simulated wave parameters
import swave_params

# Mapcube handling tools
import mapcube_tools

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

# Select the wave
example = 'no_noise'

# What type of output do we want to analyze
mctype = 'finalmaps'

# Number of trials
ntrials = 2

# Number of images
max_steps = 10

# Accumulation in the time direction
accum = 2

# Summing in the spatial directions
spatial_summing = 4

# Radii of the morphological operations
radii = [[5, 5], [11, 11], [22, 22]]

# Position measuring choices
position_choice = 'maximum'
error_choice = 'maxwidth'

# Image output directory
output = '~/eitwave/img/'

# Image output directory and filename
imgdir = os.path.expanduser(output)
filename = ''
sradii = ''
for r in radii:
    for v in r:
        sradii = sradii + str(v) + '_'
sradii = sradii[0: -1]
for loc in [example,
            mctype,
            str(ntrials) + '_' + str(max_steps) + '_' + str(accum) + '_' + str(spatial_summing),
            sradii,
            position_choice + '_' + error_choice]:
    imgdir = os.path.join(imgdir, loc)
    filename = filename + loc + '.'
filename = filename[0: -1]
if not(os.path.exists(imgdir)):
    os.makedirs(imgdir)

# Load in the wave params
params = swave_params.waves()[example]

# Storage for the results
results = []

# Go through all the test waves, and apply AWARE.
for i in range(0, ntrials):
    # Let the user which trial is happening
    print('Trial %i out of %i' % (i + 1, ntrials))

    # Simulate the wave and return a dictionary
    mc = test_wave2d.simulate_wave2d(params=params, max_steps=max_steps,
                                     verbose=True, output=[mctype])[mctype]

    # Time when we think that the event started
    originating_event_time = mc[0].date

    # Accumulate the data in space and time to increase the signal to noise
    # ratio
    mc = mapcube_tools.accumulate(mapcube_tools.superpixel(mc, (spatial_summing, spatial_summing)),
                                  accum)

    # Unravel the data
    unraveled = aware_utils.map_unravel(mc, params)

    # AWARE image processing
    umc = aware.processing(unraveled, radii=radii)

    # Get and store the dynamics of the wave front
    results.append(aware.dynamics(umc[1:],
                                  params,
                                  originating_event_time=originating_event_time,
                                  error_choice=error_choice,
                                  position_choice=position_choice))

#
# Plot out summary dynamics for all the simulated waves
#
aware_plot.swave_summary_plots(imgdir, filename, results, params)

