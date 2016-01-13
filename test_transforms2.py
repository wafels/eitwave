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

import util

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
random_seed = 42
np.random.seed(random_seed)

# Select the wave
example = 'wavenorm4_slow'
# example = 'no_noise_no_solar_rotation_slow'

# What type of output do we want to analyze
mctype = 'finalmaps'

# Number of images
max_steps = 2

# Accumulation in the time direction
accum = 2

# Summing in the spatial directions
spatial_summing = [4, 4]*u.pix

# Radii of the morphological operations
radii = [[5, 5], [11, 11], [22, 22]]

# Position measuring choices
position_choice = 'average'
error_choice = 'width'

# Degree of polynomial to fit
n_degree = 1

# Unraveling factor
unraveling_factor = 1.0

# Output directory
output = '~/eitwave/'

# Output types
otypes = ['img', 'pkl']

# RANSAC
ransac_kwargs = {"random_state": random_seed}

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


# Load in the wave params
params = swave_params.waves(lon_start=-180 * u.degree + 0.0 * u.degree)[example]

# Unraveling params are different compared to the wave definition params
params_unravel = copy.deepcopy(params)

# Sum over many of the original bins used to create the wave in an attempt to
# beat down transform artifacts
params_unravel['lon_bin'] = unraveling_factor * params['lon_bin']
params_unravel['lat_bin'] = unraveling_factor * params['lat_bin']

# Move zero location of longitudinal reconstruction relative to the
# wavefront
# params_unravel['lon_min'] = params_unravel['lon_min']
# params_unravel['lon_max'] = params_unravel['lon_max']

# Storage for the results
results = []

# Go through all the test waves, and apply AWARE.
# Let the user which trial is happening
print('\nSimulating %s ' % example)
print(' - position choice = %s' % position_choice)
print(' - error choice = %s' % error_choice)
print(' - unraveling factor = %f' % unraveling_factor)

# Simulate the wave and return a dictionary
out = test_wave2d.simulate_wave2d(params=params, max_steps=max_steps,
                                  verbose=True, output=['raw', 'transformed', 'noise', 'finalmaps'])

# Test the round trip HPC -> HG -> HPC using util.
map_index = 0
fmap = out['finalmaps'][map_index]
fmap2hg = util.map_hpc_to_hg_rotate(fmap, epi_lon=0, epi_lat=90.0, lon_bin=1.0, lat_bin=1.0)
fmap2hg2hpc = util.map_hg_to_hpc_rotate(fmap2hg, ??? )

# Difference should be zero everywhere in a perfect world!
diff_util_hpc2hg2hpc = fmap.data - fmap2hg2hpc

# Test the difference between the wave simulation HG -> HPC transform and the
# util version of the HG -> HPC transform.
rmap = out['raw'][map_index]
tmap = out['transformed'][map_index]
rmap2hpc = util.map_hg_to_hpc(rmap, ???)

# Compare the two transforms. Should be identical.
diff_due2different_transforms = tmap.data - rmap2hpc.data

