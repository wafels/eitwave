
#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import numpy as np
import astropy.units as u

observational = True

###############################################################################
#
# Observations of a wave
#

if not observational:
    # Which wave?
    # wave_name = 'lowsnr'
    # wave_name = 'lowsnr_full360'

    # wave_name = 'lowsnr_full360_slow'
    # wave_name = 'lowsnr_full360_slow_nosolarrotation'
    # wave_name = 'lowsnr_full360_slow_displacedcenter'
    # wave_name = 'lowsnr_full360_slow_nosolarrotation_displacedcenter'

    # wave_name = 'lowsnr_full360_slow_accelerated'
    # wave_name = 'lowsnr_full360_slow_nosolarrotation_accelerated'
    # wave_name = 'lowsnr_full360_slow_accelerated_displacedcenter'
    # wave_name = 'lowsnr_full360_slow_accelerated_nosolarrotation_displacedcenter'
    # wave_name = 'lowsnr_full360_slow_nosolarrotation_accelerated_displacedcenter'
    # wave_name = 'basicwave_full360_slow_displacedcenter'
    # wave_name = 'hisnr_full360_slow'
    # wave_name = "hisnr_full360_nosolarrotation_slow"
    # wave_name = 'hisnr_full360_accelerated_nosolarrotation'
    # wave_name = "hisnr_full360_slow_nosolarrotation_accelerated"
    # wave_name = "hisnr_full360_nosolarrotation_acceleration_slow2"
    # wave_name = "hisnr_full360_nosolarrotation_acceleration_slow3"
    # wave_name = 'hisnr_full360_slow_nosolarrotation_accelerated_displacedcenter'
    # wave_name = "wnbacksnr"
    # wave_name = "superlowsnr"
    # wave_name = "superlowsnr_displacedcenter"
    wave_name = "superhisnr"
    # wave_name = "superhisnr_displacedcenter"
    # If True, use pre-saved data
    use_saved = False

    # If True, save the test waves
    save_test_waves = False

    # Number of images
    max_steps = 60

else:
    # Which wave?
    wave_name = 'longetal2014_figure4'  # June 7 2011
    #wave_name = 'longetal2014_figure7'  # 13 February 2011
    #wave_name = 'longetal2014_figure8a'  # 15 February 2011
    #wave_name = 'longetal2014_figure8e'  # 16 February 2011
    #wave_name = 'longetal2014_figure6'  # 8 February 2011, no wave
    #wave_name = 'byrneetal2013_figure12'  # 16 February 2011

    # Not needed when using observed data
    use_saved = None
    save_test_waves = None
    max_steps = None

    # Root location of the test observational data
    test_observational_root = os.path.expanduser('~/Data/eitwave/test_observational_data')


# Number of trials
n_random = 100

# Reproducible randomness
random_seed = 42
np.random.seed(random_seed)

# Use the second version of the HG to HPC transform
use_transform2 = True

###############################################################################
#
# Preparation of the simulated observations to create a mapcube that will be
# used by AWARE.
#

# Analysis source data
analysis_data_sources = 'finalmaps'
#analysis_data_sources = 'transformed'
#analysis_data_sources = 'raw'
#analysis_data_sources = ('raw',)

if not observational:
    # Summing of the simulated observations in the time direction
    temporal_summing = 2

    # Summing of the simulated observations in the spatial directions
    spatial_summing = [2, 2]*u.pix
else:
    # Summing of the observations in the time direction
    temporal_summing = 2

    # Summing of the observations in the spatial directions
    spatial_summing = [2, 2]*u.pix


# Oversampling along the wavefront
along_wavefront_sampling = 1

# Oversampling perpendicular to wavefront
perpendicular_to_wavefront_sampling = 1

# Unraveling parameters used to convert HPC image data to HG data.
# There are 360 degrees in the longitudinal direction, and a maximum of 180
# degrees in the latitudinal direction.
# transform_hpc2hg_parameters = {'lon_bin': 1.0*u.degree,
#                               'lat_bin': 1.0*u.degree,
#                               'lon_num': 360*along_wavefront_sampling*u.pixel,
#                               'lat_num': 900*perpendicular_to_wavefront_sampling*u.pixel}
"""
transform_hpc2hg_parameters = {'lon_bin': 5.0*u.degree,
                               'lat_bin': 0.2*u.degree,
                               'lon_num': 361*along_wavefront_sampling*u.pixel,
                               'lat_num': 181*perpendicular_to_wavefront_sampling*u.pixel}
"""
transform_hpc2hg_parameters = {'lon_bin': 1.0*u.degree,
                               'lat_bin': 1.0*u.degree,
                               'lon_num': 360*along_wavefront_sampling*u.pixel,
                               'lat_num': 720*perpendicular_to_wavefront_sampling*u.pixel}

# HPC to HG transformation: methods used to calculate the griddata
# interpolation
# griddata_methods = ('linear', 'nearest')
griddata_methods = 'nearest'
# griddata_methods = 'linear'


###############################################################################
#
# AWARE processing: details
#

# Which version of AWARE to use?
aware_version = 0

# AWARE processing
intensity_scaling_function = np.sqrt
histogram_clip = [0.0, 99.0]


def morphology_radii(version):
    # Radii of the morphological operations in the HG co-ordinate and HPC
    # co-ordinates
    if version == 1:
        return [[5, 5]*u.degree, [11, 11]*u.degree, [22, 22]*u.degree]
    elif version == 0:
        #return [[22, 22]*u.arcsec, [44, 44]*u.arcsec, [88, 88]*u.arcsec]
        #return [[11, 11]*u.arcsec, [22, 22]*u.arcsec, [44, 44]*u.arcsec]
        #return [[11, 11]*u.arcsec]
        return [[22, 22]*u.arcsec]

# Number of longitude starting points
n_longitude_starts = 1

#  The first longitude
longitude_base = 0.0 * u.degree

# Create the longitude starts
longitude_starts = longitude_base + np.linspace(0.0, 89.0, num=n_longitude_starts) * u.degree

################################################################################
#
# Measure the velocity and acceleration of the HG arcs
#

# Position measuring choices
# position_choice = 'average'
position_choice = 'weighted_center'

#position_choice = 'gaussian'
error_choice = 'width'
#error_choice = 'std'

# Number of degrees in the polynomial fit
n_degrees = (1, 2)

# RANSAC
ransac_kwargs = {"random_state": random_seed}
#ransac_kwargs = None

error_tolerance_kwargs = {'threshold_error': np.median,
                          'function_error': np.median}

error_tolerance_kwargs = {'function_error': np.median}

fit_method = 'conditional'

great_circle_points = 100000

################################################################################
#
# Where to dump the output, and what kind of output
#

# Output directory
output = '~/eitwave/'

# Special designation: an extra description added to the file and directory
# names in order to differentiate between experiments on the same example wave.
# special_designation = '_ignore_first_six_points'
# special_designation = '_after_editing_for_dsun_and_consolidation'
# special_designation = '_fix_for_crpix12'
special_designation = ''

# Output types
otypes = ['img', 'dat']

# Is this for the AWARE paper?
for_paper = True
