
#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import numpy as np
import astropy.units as u

observational = False

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
    wave_name = 'lowsnr_full360_slow_displacedcenter'
    # wave_name = 'lowsnr_full360_slow_nosolarrotation_displacedcenter'

    # wave_name = 'lowsnr_full360_slow_accelerated'
    # wave_name = 'lowsnr_full360_slow_accelerated_nosolarrotation'
    # wave_name = 'lowsnr_full360_slow_accelerated_displacedcenter'
    # wave_name = 'lowsnr_full360_slow_accelerated_nosolarrotation_displacedcenter'

    # If True, use pre-saved data
    use_saved = False

    # If True, save the test waves
    save_test_waves = False

    # Number of trials
    ntrials = 100

    # Number of images
    max_steps = 80

else:
    # Which wave?
    wave_name = 'long_et_al_2014_figure_1'

    # Number of trials
    ntrials = 1

    # Not needed when using observed data
    use_saved = None
    save_test_waves = None
    max_steps = None


# Reproducible randomness
random_seed = 42
np.random.seed(random_seed)

#  The first longitude
longitude_start = (-180 + 0) * u.degree

# Use the second version of the HG to HPC transform
use_transform2 = True

###############################################################################
#
# Preparation of the simulated observations to create a mapcube that will be
# used by AWARE.
#

# Analysis source data
analysis_data_sources = ('finalmaps',)

# Summing of the simulated observations in the time direction
temporal_summing = 2

# Summing of the simulated observations in the spatial directions
spatial_summing = [4, 4]*u.pix

# Oversampling along the wavefront
along_wavefront_sampling = 1

# Oversampling perpendicular to wavefront
perpendicular_to_wavefront_sampling = 1

# Unraveling parameters used to convert HPC image data to HG data.
# There are 360 degrees in the longitudinal direction, and a maximum of 180
# degrees in the latitudinal direction.
transform_hpc2hg_parameters = {'lon_bin': 1.0*u.degree,
                               'lat_bin': 1.0*u.degree,
                               'lon_num': 360*along_wavefront_sampling*u.pixel,
                               'lat_num': 720*perpendicular_to_wavefront_sampling*u.pixel}

# HPC to HG transformation: methods used to calculate the griddata interpolation
griddata_methods = ('linear', 'nearest')


###############################################################################
#
# AWARE processing: details
#

# Which version of AWARE to use?
aware_version = 1

# AWARE processing
intensity_scaling_function = np.sqrt
histogram_clip = [0.0, 99.0]


def morphology_radii(version):
    # Radii of the morphological operations in the HG co-ordinate and HPC
    # co-ordinates
    if version == 1:
        return [[5, 5]*u.degree, [11, 11]*u.degree, [22, 22]*u.degree]
    elif version == 0:
        return [[22, 22]*u.arcsec, [44, 44]*u.arcsec, [88, 88]*u.arcsec]


################################################################################
#
# Measure the velocity and acceleration of the HG arcs
#

# Position measuring choices
position_choice = 'average'
error_choice = 'width'

# Number of degrees in the polynomial fit
n_degrees = (1, 2)

# RANSAC
# ransac_kwargs = {"random_state": random_seed}
ransac_kwargs = None


################################################################################
#
# Where to dump the output
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
otypes = ['img', 'pkl']
