#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# AWARE utilities
import aware_utils

# Simulated wave parameters
import swave_params


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
# example = 'wavenorm4_slow'
example = 'no_noise_no_solar_rotation_slow_360'

# Use pre-saved data
use_saved = False

# Number of trials
ntrials = 3

# Number of images
max_steps = 80

# Accumulation in the time direction
accum = 2

# Summing in the spatial directions
spatial_summing = [4, 4]*u.pix

# Radii of the morphological operations in the HG co-ordinate syste,
radii = [[5, 5]*u.degree, [11, 11]*u.degree, [22, 22]*u.degree]

# Oversampling along the wavefront
along_wavefront_sampling = 5

# Oversampling perpendicular to wavefront
perpendicular_to_wavefront_sampling = 5

# If False, load the test waves
save_test_waves = False

# Position measuring choices
position_choice = 'average'
error_choice = 'width'

# Output directory
output = '~/eitwave/'

# Output types
otypes = ['img', 'pkl']

# Analysis source data
analysis_data_sources = ('finalmaps', 'raw', 'raw_no_accumulation')
analysis_data_sources = ('finalmaps',)

# Methods used to calculate the griddata interpolation
griddata_methods = ('linear', 'nearest')


# Special designation: an extra description added to the file and directory
# names in order to differentiate between experiments on the same example wave.
#special_designation = '_ignore_first_six_points'
#special_designation = '_after_editing_for_dsun_and_consolidation'
# special_designation = '_fix_for_crpix12'
special_designation = ''

# Output directories and filename
odir = os.path.expanduser(output)
otypes_dir = {}
otypes_filename = {}

# Morphological radii
sradii = ''
for r in radii:
    for v in r:
        sradii = sradii + str(v.value) + '_'
sradii = sradii[0: -1]

# Create the storage directories and filenames
for ot in otypes:
    # root directory
    idir = os.path.join(odir, ot)

    # filename
    filename = ''

    # All the subdirectories
    for loc in [example + special_designation,
                'finalmaps',
                str(ntrials) + '_' + str(max_steps) + '_' + str(accum) + '_' + str(spatial_summing.value),
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
filepath = os.path.join(otypes_dir['pkl'], otypes_filename['pkl'] + '.pkl')
print 'Loading ' + filepath
f = open(filepath, 'rb')
results = pickle.load(f)
f.close()

# Number of trials
ntrial = len(results)

# Get the methods
methods = results[0].keys()

# How many arcs?
narc = len(results[0][methods[0]])

# Polynomial fits used
polynomials = ('linear', 'quadratic')

# Storage for the arrays
fitted = np.zeros((ntrial, narc))
v = np.zeros_like(fitted, dtype=float)
ve = np.zeros_like(fitted, dtype=float)

# Make a plot for each griddata method and polynomial fit choice
for i_method, method in enumerate(methods):
    for i_polynomial, polynomial in enumerate(polynomials):
        for i_trial, trial in enumerate(results):
            trial_by_method = trial[method]
            for i_arc, arcs in enumerate(trial_by_method):
                this_fit = arcs[i_polynomial]
                v[i_trial, i_arc] = this_fit.velocity.value if this_fit.fit_able else np.nan
                ve[i_trial, i_arc] = this_fit.velocity_error.value if this_fit.fit_able else np.nan

            vmean = np.nanmean(v, axis=0)
            vmedian = np.nanmean(v, axis=0)

"""

fitname = ['linear fit (degree 1)'] #', 'quadratic fit (degree 2)']
for offset in (0, 1):
    degree_index = np.arange(offset, 200, 2, dtype=np.int)

    results = [all_results[i] for i in degree_index]

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
    rchi2 = np.zeros_like(fitted)
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
                #a[itrial, ir] = r.acceleration.value
                #ae[itrial, ir] = r.acceleration_error.value
                rchi2[itrial, ir] = r.rchi2
            else:
                fitted[itrial, ir] = False

    #
    # Make the velocity and acceleration plots
    #
    plt.close('all')
    for j in range(0, 1):
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
        qmeane = []
        qmedian = []
        qmediane = []
        mean_index = []
        for i in range(0, narc):
            # Find where the successful fits were
            succesful_fit = fitted[:, i]

            # Reduced chi-squared
            rc2 = rchi2[:, i]

            # Successful fit
            f = succesful_fit * (rc2 < rchi2_limit)

            # Indices of the successful fits
            trialindex = np.nonzero(f)

            # Number of successful trials
            nfound[i] = np.sum(f)

            # Mean value over the successful trials
            thismean = np.sum(q[trialindex, i]) / (1.0 * nfound[i])

            # Estimated error - root mean square
            thiserror = np.sqrt(np.mean(qe[trialindex, i] ** 2))

            # Median value
            this_median = np.median(q[trialindex, i])

            # Mean absolute deviation
            mad = np.median(np.abs(q[trialindex, i] - this_median))

            if np.isfinite(thismean):
                mean_index.append(i)
                qmean.append(thismean)
                qmeane.append(thiserror)
                qmedian.append(this_median)
                qmediane.append(mad)

        # Plot the mean values found
        ax1.errorbar(mean_index, qmean, yerr=(qmeane, qmeane), linewidth=2, label='mean arc %s' % qname)
        ax1.errorbar(mean_index, qmedian, yerr=(qmediane, qmediane), linewidth=2, label='median arc %s' % qname)


        # Plot the mean of the means
        qqmean = np.asarray(qmean)
        #qqlow = np.percentile(qqmean, 2.5)
        #qqhigh = np.percentile(qqmean, 97.5)
        #qqget = (qqlow < qqmean) * (qqmean < qqhigh)
        super_mean = np.mean(qqmean)
        #super_mean_std = np.std(qqmean[qqget])
        ax1.axhline(super_mean, color='k', linestyle='--', linewidth=2, label='mean %s across wavefront = %f' % (qname, super_mean))
        #ax1.axhline(super_mean + super_mean_std, color='k', linewidth=2, linestyle=':', label='super mean %s + std = %f' % (qname, super_mean + super_mean_std))
        #ax1.axhline(super_mean - super_mean_std, color='k', linewidth=2, linestyle=':', label='super mean %s - std = %f' % (qname, super_mean - super_mean_std))
        qqmedian = np.asarray(qmedian)
        super_median = np.median(qqmedian)
        ax1.axhline(super_median, color='k', linestyle='-.', linewidth=2, label='median %s across wavefront = %f' % (qname, super_median))

        # Plot the line that indicates the true velocity at t=0
        ax1.axhline(initial_value, color='k',  linewidth=4, label='true %s=%f%s' % (qname, initial_value, qunit))

        # Labelling the quantity plot
        xlabel = 'arc index'
        ylabel = 'estimated %s (%s)' % (qname, qunit)
        title = '%s: %s: estimated %s across wavefront' % (fitname[offset], qname, params['name'])
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(title)
        ax1.set_ylim(430, 470)
        ax1.legend(framealpha=0.9, loc=4)
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
"""
