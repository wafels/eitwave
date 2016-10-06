#
# Shows some steps in the aware processing
#

import pickle
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sunpy.map
import swave_study as sws
import aware_utils
import aware3

image_filepath = '~/projects/eitwave-paper/show_aware_processing.eps'

info = {'longetal2014_figure8a': 20,
        'longetal2014_figure8e': 20,
        'longetal2014_figure4': 20}

fontsize = 12
maps = {}

###############################################################################
#
# Setting up the wave data
#
# Observational or simulated data?
observational = sws.observational

# Which wave?
wave_name = sws.wave_name

# If True, use pre-saved data
use_saved = sws.use_saved

# If True, save the test waves
save_test_waves = sws.save_test_waves

# Number of trials
n_random = sws.n_random

# Number of images
max_steps = sws.max_steps

# Reproducible randomness
np.random.seed(sws.random_seed)

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

# Oversampling along the wavefront
along_wavefront_sampling = sws.along_wavefront_sampling

# Oversampling perpendicular to wavefront
perpendicular_to_wavefront_sampling = sws.perpendicular_to_wavefront_sampling

# Unraveling parameters used to convert HPC image data to HG data.
# There are 360 degrees in the longitudinal direction, and a maximum of 180
# degrees in the latitudinal direction.
transform_hpc2hg_parameters = sws.transform_hpc2hg_parameters

# HPC to HG transformation: methods used to calculate the griddata interpolation
griddata_methods = sws.griddata_methods


###############################################################################
#
# AWARE processing: details
#

# Which version of AWARE to use?
aware_version = sws.aware_version

# AWARE processing
intensity_scaling_function = sws.intensity_scaling_function
histogram_clip = sws.histogram_clip

# Radii of the morphological operations in the HG co-ordinate and HPC
# co-ordinates
radii = sws.morphology_radii(aware_version)

# Number of longitude starting points
longitude_starts = sws.longitude_starts


###############################################################################
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

# Error tolerance keywords
error_tolerance_kwargs = sws.error_tolerance_kwargs

# Fit method
fit_method = sws.fit_method

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

# Interactive show for speed
plt.ion()

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
    for loc in [wave_name + special_designation,
                'use_transform2=' + str(use_transform2),
                'finalmaps',
                str(n_random) + '_' + str(max_steps) + '_' + str(temporal_summing) + '_' + str(spatial_summing.value),
                sradii,
                position_choice + '_' + error_choice,
                str(ransac_kwargs)]:
        idir = os.path.join(idir, loc)
        filename = filename + loc + '.'
    filename = filename[0: -1]
    if not(os.path.exists(idir)):
        os.makedirs(idir)
    otypes_dir[ot] = idir
    otypes_filename[ot] = filename

#
# Load in data
#
index = 20
create = False

if create:
    mc = aware_utils.create_input_to_aware_for_test_observational_data(wave_name)['finalmaps']
    develop = {'img': os.path.join(otypes_dir['img'], otypes_filename['img']),
               'dat': os.path.join(otypes_dir['dat'], otypes_filename['dat'])}
    aware_processed, develop_filepaths = aware3.processing(mc,
                                                        develop=develop,
                                                        radii=radii,
                                                        func=intensity_scaling_function,
                                                        histogram_clip=histogram_clip)
else:
    print('Loading datasets.')
    develop_filepaths = {}
    root = os.path.join(otypes_dir['dat'], otypes_filename['dat'])
    develop_filepaths['rdpi_mc'] = root + "_rdpi_mc.pkl"
    develop_filepaths['np_median_dc'] = root + "_np_median_dc_0.npy"
    develop_filepaths['np_meta'] = root + "_np_meta.pkl"
    develop_filepaths['np_nans'] = root + "_np_nans.npy"
    develop_filepaths['np_closing_dc'] = root + "_np_closing_dc_0.npy"

#
# Load in the running difference of persistence images.
#
f = open(develop_filepaths['rdpi_mc'], 'rb')
rdpi_mc = pickle.load(f)
f.close()

#
# Load in the median cleaning data and create a map.
#
f = open(develop_filepaths['np_median_dc'], 'rb')
np_median_dc = np.load(f)
f.close()

f = open(develop_filepaths['np_meta'], 'rb')
np_meta = pickle.load(f)
f.close()

f = open(develop_filepaths['np_nans'], 'rb')
np_nans = np.load(f)
f.close()

#
# Load in the data after the closing operation has been applied.
#
f = open(develop_filepaths['np_closing_dc'], 'rb')
np_closing_dc = np.load(f)
f.close()

m_rdpi = rdpi_mc[index]
m_rdpi.plot_settings['cmap'] = cm.gray_r
m_median = sunpy.map.Map(np_median_dc[:, :, index], np_meta[index])
m_median.plot_settings['cmap'] = cm.gray_r
m_closing = sunpy.map.Map(np_closing_dc[:, :, index], np_meta[index])
m_closing.plot_settings['cmap'] = cm.gray_r

#
# Make the plot
#
title = ['RDP', 'after median filter', 'after closing']
plt.close('all')
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

for i, m in enumerate([m_rdpi, m_median, m_closing]):
    ta = axes[i]
    m.plot(axes=ta, title=title[i])
    m.draw_limb(color='red')
    ta.set_xlabel('x (arcsec)', fontsize=fontsize)
    xtl = ta.axes.xaxis.get_majorticklabels()
    for l in range(0, len(xtl)):
        xtl[l].set_fontsize(0.67*fontsize)
    ta.set_ylabel('y (arcsec)', fontsize=fontsize)
    ytl = ta.axes.yaxis.get_majorticklabels()
    for l in range(0, len(ytl)):
        ytl[l].set_fontsize(0.67*fontsize)

plt.tight_layout()
plt.show()
plt.savefig(os.path.expanduser(image_filepath))
plt.close('all')
