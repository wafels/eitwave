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
from aware_constants import solar_circumference_per_degree_in_km

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
use_error_bar = False

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

# Great circle points
great_circle_points = sws.great_circle_points

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
    otypes_filename[ot] = filename + '.' + str(great_circle_points)


# Load in the wave params
params = swave_params.waves()[example]

#
# Load the results
#
if not os.path.exists(otypes_dir['dat']):
    os.makedirs(otypes_dir['dat'])
filepath = os.path.join(otypes_dir['dat'], otypes_filename['dat'] + '.pkl')
print('\nLoading ' + filepath + '\n')
f = open(filepath, 'rb')
results = pickle.load(f)
f.close()

# How many arcs?
nlon = len(results[0])
angles = ((np.linspace(0, 2*np.pi, nlon+1))[0:-1] * u.rad).to(u.deg)


# Initial value to the velocity
velocity_unit = u.km/u.s
acceleration_unit = u.km/u.s/u.s
true_values = {"velocity": (params['speed'][0] * aware_constants.solar_circumference_per_degree).to(velocity_unit).value,
               "acceleration": (params['acceleration'] * aware_constants.solar_circumference_per_degree).to(acceleration_unit).value}

true_value_labels = {"velocity": "km/s", "acceleration": "km/s/s"}


def extract(results, n_degree=1, measurement_type='velocity'):
    """
    Extract the particular measurements from the results structure
    :param results:
    :param n_degree:
    :param measurement_type:
    :return:
    """
    n_trials = len(results)
    nlon = len(results[0])
    measurement = np.zeros(shape=(n_trials, nlon))
    measurement_error = np.zeros_like(measurement)
    fitted = np.zeros_like(measurement)
    rchi2 = np.zeros_like(measurement)
    for this_arc in range(0, nlon):
        for this_trial in range(0, n_trials):
            answer = (results[this_trial][this_arc][n_degree-1]).answer

            if answer.fitted:
                fitted[this_trial, this_arc] = True
                rchi2[this_trial, this_arc] = answer.rchi2
                if measurement_type == 'velocity':
                    measurement[this_trial, this_arc] = (answer.velocity * solar_circumference_per_degree_in_km).value
                    measurement_error[this_trial, this_arc] = (answer.velocity_error * solar_circumference_per_degree_in_km).value
                if measurement_type == 'acceleration':
                    measurement[this_trial, this_arc] = (answer.acceleration * solar_circumference_per_degree_in_km).value
                    measurement_error[this_trial, this_arc] = (answer.acceleration * solar_circumference_per_degree_in_km).value
            else:
                fitted[this_trial, this_arc] = False
                measurement[this_trial, this_arc] = np.nan
                measurement_error[this_trial, this_arc] = np.nan
                rchi2[this_trial, this_arc] = np.nan

    return fitted, rchi2, measurement, measurement_error


def summarize(fitted, rchi2, measurement, rchi2_limit=1.5):
    """
    Create summaries of the input measurement

    :param fitted:
    :param rchi2:
    :param measurement:
    :param summary:
    :param rchi2_limit:
    :return:
    """
    nlon = measurement.shape[1]

    mean = np.zeros(shape=nlon)
    std = np.zeros_like(mean)
    median = np.zeros_like(mean)
    mad = np.zeros_like(mean)
    n_found = np.zeros_like(mean)

    for i in range(0, nlon):
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

        m = measurement[trial_index, i]

        mean[i] = np.sum(m) / (1.0 * n_found[i])

        std[i] = np.std(m)

        median[i] = np.median(m)

        mad[i] = np.median(np.abs(m - median[i]))

    mean_mean = np.mean(mean)
    mean_std = np.mean(std)
    median_median = np.median(median)
    median_mad = np.median(mad)

    return ("mean, STD", mean, std, mean_mean, mean_std),\
           ("median, MAD", median, mad, median_median, median_mad)


for n_degree in [1, 2]:

    if n_degree == 1:
        measurement_types = ['velocity']
        fit = 'linear fit'
    if n_degree == 2:
        measurement_types = ['velocity', 'acceleration']
        fit = 'quadratic fit'

    for measurement_type in measurement_types:
        true_value = true_values[measurement_type]
        true_value_label = true_value_labels[measurement_type]

        # Make plots of the central tendency of the velocity

        fitted, rchi2, measurement, measurement_error = extract(results,
                                                                n_degree=n_degree,
                                                                measurement_type=measurement_type)

        summaries = summarize(fitted, rchi2, measurement, rchi2_limit=rchi2_limit)

        for summary in summaries:
            plt.close('all')
            fig, ax = plt.subplots()
            ax.errorbar(angles.value, summary[1], summary[2], linewidth=0.5)

            ax.axhline(true_value, label="true {:s}".format(measurement_type), color='k')
            ax.set_xlabel('longitude (degrees)')
            ax.set_ylabel(measurement_type + " ({:s})".format(true_value_label))
            ax.set_title("{:s} ({:s})\n{:s}".format(measurement_type, summary[0], fit))
            ax.legend(framealpha=0.5)
            filename = otypes_filename["img"] + '.' + measurement_type + '.' + summary[0] + '.' + fit + '.png'
            file_path = os.path.join(otypes_dir['img'], filename)
            print('Saving {:s}'.format(file_path))
            fig.tight_layout()
            fig.savefig(file_path)


