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
example = 'wavenorm4_slow_displaced'

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

# Output directory
output = '~/eitwave/'

# Output types
otypes = ['img', 'pkl']

# Special designation: an extra description added to the file and directory
# names in order to differentiate between experiments on the same example wave.
special_designation = ''

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

#
# Load the results
#
if not os.path.exists(otypes_dir['pkl']):
    os.makedirs(otypes_dir['pkl'])
filepath = os.path.join(otypes_dir['pkl'], otypes_filename['pkl'] + '.pkl' )
f = open(filepath, 'rb')
results = pickle.load(f)
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

    """
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
    """

    # Mean quantity over all the trials
    qmean = []
    qmeane = []
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

        # Estimated error - root mean square
        thiserror = np.sqrt(np.mean(qe[trialindex, i] ** 2))

        if np.isfinite(thismean):
            mean_index.append(i)
            qmean.append(thismean)
            qmeane.append(thiserror)

    # Plot the mean values found
    ax1.errorbar(mean_index, qmean, yerr=(qmeane, qmeane), linewidth=2, label='mean %s' % qname)

    # Plot the mean of the means
    qqmean = np.asarray(qmean)
    qqlow = np.percentile(qqmean, 2.5)
    qqhigh = np.percentile(qqmean, 97.5)
    qqget = (qqlow < qqmean) * (qqmean < qqhigh)
    super_mean = np.mean(qqmean[qqget])
    super_mean_std = np.std(qqmean[qqget])
    ax1.axhline(super_mean, color='k', linewidth=2, label='super mean %s = %f' % (qname, super_mean))
    ax1.axhline(super_mean + super_mean_std, color='k', linewidth=2, linestyle=':', label='super mean %s + std = %f' % (qname, super_mean + super_mean_std))
    ax1.axhline(super_mean - super_mean_std, color='k', linewidth=2, linestyle=':', label='super mean %s - std = %f' % (qname, super_mean - super_mean_std))


    # Plot the line that indicates the true velocity at t=0
    ax1.axhline(initial_value, color='k', linewidth=4, label='true %s=%f%s' % (qname, initial_value, qunit))

    # Labelling the quantity plot
    xlabel = 'arc index'
    ylabel = 'estimated %s (%s)' % (qname, qunit)
    title = '%s: estimated %s across wavefront' % (qname, params['name'])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend(framealpha=0.9)
    for tl in ax1.get_yticklabels():
        tl.set_color(qcolor)

    # Add in some 45 degree lines
    for i in all_arcindex.tolist():
        if np.mod(i, 45) == 0:
            ax1.axvline(i, color='k')
    plt.show()

# Fraction found
plt.figure(2)
plt.xlabel(xlabel)
plt.ylabel('fraction of trials with successful fit (%i trials)' % ntrials)
plt.title('%s - fraction of trials with successful fit' % params['name'])
plt.plot(all_arcindex, nfound / (1.0 * ntrials), label='fraction fitted')
plt.show()


# Add up all the fitted data that we have found and plot out the average.  Note
# that the graph below is probably most useful for circular test waves centered
# at (0,0) on the Sun.
nlon = len(dynamics)
nt = len(dynamics[0][0].pos)
arc_mask = np.zeros((ntrial, nlon, nt))
all_arcs = np.zeros_like(arc_mask)
plt.figure(3)
for itrial, dynamics in enumerate(results):

    # Go through each arc and get the results
    for lon, result in enumerate(dynamics):
        r = result[0]
        arc_mask[itrial, lon, :] = r.defined[:]
        all_arcs[itrial, lon, :] = r.pos[:]
        plt.errorbar(np.arange(0, nlon)[r.defined], r.pos[r.defined], yerr=(r.error[r.defined], r.error[r.defined]), fmt=fmt)

average_arc = np.sum(all_arcs * arc_mask, axis=(0, 1)) / np.sum(arc_mask, axis=(0, 1))

plt.plot(average_arc, linewidth=4)
plt.xlabel(r"time (in units of %s$\times$12 seconds)." % accum)
plt.ylabel("average measured angular displacement")
plt.title("%i trials, %i arcs" % (ntrial, nlon))
plt.show()

