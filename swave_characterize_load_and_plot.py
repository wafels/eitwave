#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# AWARE constants
import aware_constants

# AWARE utilities
import aware_utils

# Simulated wave parameters
import swave_params

#
import swave_study as sws

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


# Define the Analysis object
class Analysis:
    def __init__(self):
        self.method = None
        self.n_degree = None
        self.lon = None
        self.ils = None
        self.answer = None
        self.aware_version = None


###############################################################################
#
# How to select good arcs
#

# Reduced chi-squared must be below this limit
rchi2_limit = 1.0 #sws.rchi2_limit


###############################################################################
#
# Simulated observations of a wave
#
example = sws.wave_name

# Number of trials
ntrials = sws.n_random

# Number of images
max_steps = sws.max_steps

# Use the second version of the HG to HPC transform
use_transform2 = sws.use_transform2

###############################################################################
#
# Preparation of the simulated observations to create a mapcube that will be
# used by AWARE.
#

# Analysis source data
analysis_data_sources = sws.analysis_data_sources

# Summing of the simulated observations in the time direction
temporal_summing = sws.temporal_summing

# Summing of the simulated observations in the spatial directions
spatial_summing = sws.spatial_summing

# HPC to HG transformation: methods used to calculate the griddata interpolation
griddata_methods = sws.griddata_methods


###############################################################################
#
# AWARE processing: details
#

# Which version of AWARE to use?
aware_version = sws.aware_version

# Radii of the morphological operations in the HG co-ordinate and HPC
# co-ordinates
radii = sws.morphology_radii(aware_version)


################################################################################
#
# Measure the velocity and acceleration of the HG arcs
#

# Position measuring choices
position_choice = sws.position_choice
error_choice = sws.error_choice

# Number of degrees in the polynomial fit
n_degrees = sws.n_degrees

# RANSAC
ransac_kwargs = sws.ransac_kwargs


################################################################################
#
# Where to dump the output
#

# Output directory
output = sws.output

# Special designation: an extra description added to the file and directory
# names in order to differentiate between experiments on the same example wave.
# special_designation = '_ignore_first_six_points'
# special_designation = '_after_editing_for_dsun_and_consolidation'
# special_designation = '_fix_for_crpix12'
special_designation = sws.special_designation

# Output types
otypes = sws.otypes


###############################################################################
###############################################################################
#
# Everything below here is set from above
#

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
                'use_transform2=' + str(use_transform2),
                'finalmaps',
                str(ntrials) + '_' + str(max_steps) + '_' + str(temporal_summing) + '_' + str(spatial_summing.value),
                sradii,
                position_choice + '_' + error_choice,
                aware_utils.convert_dict_to_single_string(ransac_kwargs)]:
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
if not os.path.exists(otypes_dir['dat']):
    os.makedirs(otypes_dir['dat'])
filepath = os.path.join(otypes_dir['dat'], otypes_filename['dat'] + '.pkl')
filepath = '/home/ireland/eitwave/dat/hisnr_full360_nosolarrotation_acceleration_slow2_keep/use_transform2=True/finalmaps/100_80_1_[ 2.  2.]/11.0_11.0/weighted_center_width/random_state=42/hisnr_full360_nosolarrotation_acceleration_slow2.use_transform2=True.finalmaps.100_80_1_[ 2.  2.].11.0_11.0.weighted_center_width.random_state=42.pkl'
print('\nLoading ' + filepath + '\n')
f = open(filepath, 'rb')
results = pickle.load(f)
f.close()

# Number of trials
n_trials = len(results)

# How many arcs?
n_arcs = len(results[0])

# Griddata method
this_method = results[0][0][0].method

# Storage for the arrays
fitted = np.zeros((n_trials, n_arcs))
v = np.zeros_like(fitted, dtype=float)
ve = np.zeros_like(fitted, dtype=float)
a = np.zeros_like(fitted, dtype=float)
ae = np.zeros_like(fitted, dtype=float)
rchi2 = np.zeros_like(fitted, dtype=float)
n_found = np.zeros(n_arcs)

# Initial value to the velocity
velocity_unit = u.km/u.s
v_initial_value = (params['speed'][0] * aware_constants.solar_circumference_per_degree).to(velocity_unit).value
acceleration_unit = u.km/u.s/u.s
a_initial_value = (params['acceleration'] * aware_constants.solar_circumference_per_degree).to(acceleration_unit).value

# Velocity plot limits
v_ylim = [0.92*v_initial_value, 1.08*v_initial_value]

# Make a plot for each griddata method and polynomial fit choice
all_longitude = []
for this_arc in range(0, n_arcs):
    for this_trial in range(0, n_trials):
        analysis_linear, analysis_quadratic = results[this_trial][this_arc]
        ala = analysis_linear.answer

        if ala.fitted:
            fitted[this_trial, this_arc] = True
            v[this_trial, this_arc] = ala.velocity.value
            ve[this_trial, this_arc] = ala.velocity_error.value
            rchi2[this_trial, this_arc] = ala.rchi2
        else:
            fitted[this_trial, this_arc] = False
            v[this_trial, this_arc] = np.nan
            ve[this_trial, this_arc] = np.nan
            rchi2[this_trial, this_arc] = np.nan

        aqa = analysis_quadratic.answer
        if aqa.fitted:
            fitted[this_trial, this_arc] = True
            v[this_trial, this_arc] = aqa.velocity.value
            ve[this_trial, this_arc] = aqa.velocity_error.value
            a[this_trial, this_arc] = aqa.acceleration.value
            ae[this_trial, this_arc] = aqa.acceleration_error.value
            rchi2[this_trial, this_arc] = aqa.rchi2
        else:
            fitted[this_trial, this_arc] = False
            v[this_trial, this_arc] = np.nan
            ve[this_trial, this_arc] = np.nan
            a[this_trial, this_arc] = np.nan
            ae[this_trial, this_arc] = np.nan
            rchi2[this_trial, this_arc] = np.nan

    if ala.arc_identity is not None:
        longitude_unit = u.degree
        all_longitude.append(ala.arc_identity.to(longitude_unit).value)
    else:
        longitude_unit = 'index'
        all_longitude.append(this_arc)

v = v * aware_constants.solar_circumference_per_degree.value / 1000.0
ve = ve * aware_constants.solar_circumference_per_degree.value / 1000.0
a = a * aware_constants.solar_circumference_per_degree.value / 1000.0
ae = ae * aware_constants.solar_circumference_per_degree.value / 1000.0

xlabel = r'longitude ({:s}), range={:f}$\rightarrow${:f}'.format(str(longitude_unit), all_longitude[0], all_longitude[-1])

for polynomial in ('linear', 'quadratic'):
    # Now create summaries of trials
    if polynomial == 'linear' or polynomial == 'quadratic':
        mean_index = []
        q_mean = []
        q_mean_error = []
        q_median = []
        q_median_mad = []
        longitude = []
        for i in range(0, n_arcs):
            # Find where the successful fits were
            successful_fit = fitted[:, i]

            # Reduced chi-squared
            rc2 = rchi2[:, i]

            # Successful fit
            f = successful_fit * (rc2 < rchi2_limit)

            # Indices of the successful fits
            trial_index = np.nonzero(f)

            # Number of successful trials
            n_found[i] = np.sum(f)

            # Mean value over the successful trials
            v_this_mean = np.sum(v[trial_index, i]) / (1.0 * n_found[i])

            # Estimated error - root mean square
            v_this_error = np.sqrt(np.mean(ve[trial_index, i] ** 2))

            # Median value
            v_this_median = np.median(v[trial_index, i])

            # Mean absolute deviation
            v_mad = np.median(np.abs(v[trial_index, i] - v_this_median))

            if np.isfinite(v_this_mean):
                longitude.append(all_longitude[i])
                mean_index.append(i)
                q_mean.append(v_this_mean)
                q_mean_error.append(v_this_error)
                q_median.append(v_this_median)
                q_median_mad.append(v_mad)

        for velocity_assessment in ('median velocity', 'mean velocity'):
            title = 'wave = {:s}\n griddata_method = {:s}\n fit = {:s}\n assessment = {:s}'.format(example, this_method, polynomial, velocity_assessment)

            ylabel = '{:s} ({:s})'.format(velocity_assessment, str(velocity_unit))

            # Make plots of the central tendency of the velocity
            plt.close('all')
            fig, ax = plt.subplots()
            if velocity_assessment == 'mean velocity':
                ax.errorbar(longitude, q_mean, yerr=q_mean_error, label='mean velocity (std)')
            if velocity_assessment == 'median velocity':
                ax.errorbar(longitude, q_median, yerr=q_median_mad, label='median velocity (MAD)')
            ax.axhline(v_initial_value, label='true velocity ({:f} {:s})'.format(v_initial_value, str(velocity_unit)), color='k')
            ax.set_xlim(all_longitude[0], all_longitude[-1])
            #ax.set_ylim(v_ylim)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(framealpha=0.5)
            directory = otypes_dir['img']
            filename = '{:s}-gm={:s}-fit={:s}-{:s}.png'.format(otypes_filename['img'], this_method, polynomial, velocity_assessment)
            file_path = os.path.join(directory, filename)
            print('Saving {:s}'.format(file_path))
            fig.tight_layout()
            fig.savefig(file_path)

    if polynomial == 'quadratic':
        mean_index = []
        q_mean = []
        q_mean_error = []
        q_median = []
        q_median_mad = []
        longitude = []
        for i in range(0, n_arcs):
            # Find where the successful fits were
            successful_fit = fitted[:, i]

            # Reduced chi-squared
            rc2 = rchi2[:, i]

            # Successful fit
            f = successful_fit * (rc2 < rchi2_limit)

            # Indices of the successful fits
            trial_index = np.nonzero(f)

            # Number of successful trials
            n_found[i] = np.sum(f)

            # Mean value over the successful trials
            a_this_mean = np.sum(a[trial_index, i]) / (1.0 * n_found[i])

            # Estimated error - root mean square
            a_this_error = np.sqrt(np.mean(ae[trial_index, i] ** 2))

            # Median value
            a_this_median = np.median(a[trial_index, i])

            # Mean absolute deviation
            a_mad = np.median(np.abs(a[trial_index, i] - a_this_median))

            if np.isfinite(a_this_mean):
                longitude.append(all_longitude[i])
                mean_index.append(i)
                q_mean.append(a_this_mean)
                q_mean_error.append(a_this_error)
                q_median.append(a_this_median)
                q_median_mad.append(a_mad)

        # Plot the acceleration where appropriate
        for acceleration_assessment in ('mean acceleration', 'median acceleration'):
            title = 'wave = {:s}\n griddata_method = {:s}\n fit = {:s}\n assessment = {:s}'.format(example, this_method, polynomial, acceleration_assessment)
            ylabel = '{:s} ({:s})'.format(acceleration_assessment, str(acceleration_unit))
            # Make plots of the central tendency of the acceleration
            plt.close('all')
            if acceleration_assessment == 'mean acceleration':
                plt.errorbar(longitude, q_mean, yerr=q_mean_error, label='mean acceleration (std)')
            if acceleration_assessment == 'median acceleration':
                plt.errorbar(longitude, q_median, yerr=q_median_mad, label='median acceleration (MAD)')
            plt.axhline(a_initial_value, label='true acceleration ({:f} {:s})'.format(a_initial_value, str(acceleration_unit)), color='k')
            plt.xlim(all_longitude[0], all_longitude[-1])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend(framealpha=0.5)
            directory = otypes_dir['img']
            filename = '{:s}-gm={:s}-fit={:s}-{:s}.png'.format(otypes_filename['img'], this_method, polynomial, acceleration_assessment)
            file_path = os.path.join(directory, filename)
            print('Saving {:s}'.format(file_path))
            plt.tight_layout()
            plt.savefig(file_path)

# Plot the number of successful fits at each point
title = 'wave = {:s}\n griddata_method = {:s}\n fit = {:s}\n'.format(example, this_method, polynomial)
plt.close('all')
plt.plot(all_longitude, n_found, label='qualifying fits')
plt.axhline(n_trials, label='number of trials', color='k')
plt.xlim(all_longitude[0], all_longitude[-1])
plt.xlabel(xlabel)
plt.ylabel('# qualifying fits')
plt.ylim(0, 1.05*n_trials)
plt.title(title)
plt.legend(framealpha=0.5, loc='lower right')
directory = otypes_dir['img']
filename = '{:s}-gm={:s}-fit={:s}-{:s}.png'.format(otypes_filename['img'], this_method, polynomial, 'fitted')
file_path = os.path.join(directory, filename)
print('Saving {:s}'.format(file_path))
plt.tight_layout()
plt.savefig(file_path)

"""
Notes
-----

Make scatter plots of velocity versus acceleration.

"""
