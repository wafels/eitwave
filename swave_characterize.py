#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

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
example = 'basic_wave'

# What type of output do we want to analyze
mctype = 'finalmaps'

# Number of trials
ntrials = 2

# Number of images
max_steps = 80

# Accumulation in the time direction
accum = 2

# Summing in the spatial directions
spatial_summing = 4

# Radii of the morphological operations
radii = [[5, 5], [11, 11], [22, 22]]

# Position measuring choices
position_choice = 'maximum'
error_choice = 'maxwidth'

# Output directory
output = '~/eitwave/'

# Output types
otypes = ['img', 'pkl']

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
    for loc in [example,
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
    results.append(aware.dynamics(umc,
                                  params,
                                  originating_event_time=originating_event_time,
                                  error_choice=error_choice,
                                  position_choice=position_choice,
                                  returned=['answer']))
#
# Save the results
#
if not os.path.exists(otypes_dir['pkl']):
    os.makedirs(otypes_dir['pkl'])
filepath = os.path.join(otypes_dir['pkl'], otypes_filename['pkl'] + '.pkl' )
f = open(filepath, 'wb')
pickle.dump(results, f)
f.close()

# Number of trials
ntrial = len(results)

# Number of arcs
narc = len(results[0])

# Storage for the results
fitted = np.zeros((ntrial, narc))
v = np.zeros_like(fitted)
ve = np.zeros_like(fitted)
a = np.zeros_like(fitted)
ae = np.zeros_like(fitted)
nfound = np.zeros(narc)

# Indices of all the arcs
all_arcindex = np.arange(0, narc)

# Quantity plots
qcolor = 'r'
fmt = qcolor + 'o'

# Number of trials with a successful fit.
nfcolor = 'b'

#
# Recover the information we need
#
for itrial, dynamics in enumerate(results):

    # Go through each arc and get the results
    for ir, rr in enumerate(dynamics):
        r = rr[0]
        if r.fitted:
            fitted[itrial, ir] = True
            v[itrial, ir] = r.velocity.value
            ve[itrial, ir] = r.velocity_error.value
            a[itrial, ir] = r.acceleration.value
            ae[itrial, ir] = r.acceleration_error.value
        else:
            fitted[itrial, ir] = False

#
# Make the velocity and acceleration plots
#
plt.close('all')
for j in range(0, 2):
    fig, ax1 = plt.subplots()

    # Select which quantity to plot
    if j == 0:
        q = v
        qe = ve
        qname = 'velocity'
        qunit = 'km/s'
        initial_value = (params['speed'][0] * aware_utils.solar_circumference_per_degree).to(qunit).value
    else:
        q = a
        qe = ae
        qname = 'acceleration'
        qunit = 'm/s2'
        initial_value = (params['speed'][1] * aware_utils.solar_circumference_per_degree / u.s).to(qunit).value

    # Initial values to get the plot legend labels done
    arcindex = np.nonzero(fitted[0, :])[0].tolist()
    qerr = qe[0, arcindex]
    ax1.errorbar(arcindex, q[0, arcindex], yerr=(qerr, qerr),
                 fmt=fmt, label='estimated %s' % qname)
    # Plot the rest of the values found.
    if ntrial > 1:
        for i in range(1, ntrial):
            arcindex = np.nonzero(fitted[i, :])[0].tolist()
            qerr = qe[i, arcindex]
            ax1.errorbar(arcindex, q[i, arcindex], yerr=(qerr, qerr), fmt=fmt)

    # Mean quantity over all the trials
    qmean = []
    mean_index = []
    for i in range(0, narc):
        # Find where the successful fits were
        f = fitted[:, i]

        # Indices of the successful fits
        trialindex = np.nonzero(f)

        # Number of successful trials
        nfound[i] = np.sum(f)

        # Mean value over the successful trials
        thismean = np.sum(q[trialindex, i]) / (1.0 * nfound[i])

        if np.isfinite(thismean):
            mean_index.append(i)
            qmean.append(thismean)

    # Plot the mean values found
    ax1.plot(mean_index, qmean, label='mean %s' % qname)

    # Plot the line that indicates the true velocity at t=0
    ax1.axhline(initial_value, color=qcolor, linewidth=2, label='true %s=%f%s' % (qname, initial_value, qunit))

    # Labelling the quantity plot
    ax1.set_xlabel('arc index')
    ax1.set_ylabel('estimated %s (km/s)' % qname)
    ax1.legend(framealpha=0.5)
    ax1.set_title('%s: estimated %s across wavefront' % (qname, params['name']))
    for tl in ax1.get_yticklabels():
        tl.set_color(qcolor)

    # Plot the fraction
    #ax2 = ax1.twinx()
    #ax2.plot(all_arcindex, nfound / np.float64(ntrial),
    #         label='fraction of trials fitted', color=nfcolor)
    #ax2.set_ylabel('fraction of trials fitted [%i trials]' % ntrial,
    #               color=nfcolor)
    #for tl in ax2.get_yticklabels():
    #    tl.set_color(nfcolor)

    # Save the figure
    #plt.savefig(os.path.join(imgdir, '%s_initial_%s.png' % (filename, qname)))
    plt.show()


#
# Plot out summary dynamics for all the simulated waves
#
#aware_plot.swave_summary_plots(imgdir, filename, results, params)

