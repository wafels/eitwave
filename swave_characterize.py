#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

#
plt.ion()

#
from sunpy.map import Map

# Main AWARE processing and detection code
import aware

# AWARE utilities
import util

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
#example = 'no_noise_no_solar_rotation_slow_360'

# What type of output do we want to analyze
mctype = 'finalmaps'

# Use pre-saved data
use_saved = False

# Number of trials
ntrials = 1

# Number of images
max_steps = 80

# Accumulation in the time direction
accum = 2

# Summing in the spatial directions
spatial_summing = [4, 4]*u.pix

# Radii of the morphological operations
radii = [[5, 5]*u.degree, [11, 11]*u.degree, [22, 22]*u.degree]

# Oversampling along the wavefront
along_wavefront_sampling = 1

# Oversampling perpendicular to wavefront
perpendicular_to_wavefront_sampling = 1

# If False, load the test waves
save_test_waves = False

# Position measuring choices
position_choice = 'average'
error_choice = 'width'

# Degree of polynomial to fit
n_degree = 1

# Output directory
output = '~/eitwave/'

# Output types
otypes = ['img', 'pkl']

# RANSAC
ransac_kwargs = {"random_state": random_seed}

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
                mctype,
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
params = swave_params.waves(lon_start=-180 * u.degree + 10 * u.degree)[example]

# Unraveling parameters used to convert HPC image data to HG data.  Trying
# oversampling in order to get a good sampling on the wavefront.  Then use
# superpixels of this transformed data to get the final HG image we will use to
# do the wave detection.
unraveling_hpc2hg_parameters = {'lon_bin': 1.0*u.degree,
                                'lat_bin': 1.0*u.degree,
                                'epi_lon': 0.0*u.degree,
                                'epi_lat': 0.0*u.degree,
                                'lon_num': 360*along_wavefront_sampling*u.pixel,  # Sample heavily across the wavefront
                                'lat_num': 360*perpendicular_to_wavefront_sampling*u.pixel}

# Storage for the results
results = []

# Go through all the test waves, and apply AWARE.
for i in range(0, ntrials):
    # Let the user which trial is happening
    print('\nSimulating %s ' % example)
    print(' - special designation = %s' % special_designation)
    print(' - position choice = %s' % position_choice)
    print(' - error choice = %s' % error_choice)
    print(' - along wavefront sampling = %i' % along_wavefront_sampling)
    print(' - perpendicular to wavefront sampling = %i' % perpendicular_to_wavefront_sampling)
    print(' - starting trial %i out of %i\n' % (i + 1, ntrials))

    if not use_saved:
        # Simulate the wave and return a dictionary
        print(" - Creating test waves.")
        out = test_wave2d.simulate_wave2d(params=params,
                                          max_steps=max_steps,
                                          verbose=True,
                                          output=['raw', 'transformed', 'noise', 'finalmaps'])
        if save_test_waves:
            print(" - Saving test waves.")
            file_path = os.path.join(otypes_dir['pkl'], otypes_filename['pkl'] + '.pkl')
            print('Saving to %s' % file_path)
            f = open(file_path, 'wb')
            pickle.dump(out, f)
            f.close()
    else:
        print(" - Loading test waves.")
        file_path = os.path.join(otypes_dir['pkl'], otypes_filename['pkl'] + '.pkl')
        print('Loading from %s' % file_path)
        f = open(file_path, 'rb')
        out = pickle.load(f)
        f.close()

    # Get the final map out
    mc = out['finalmaps']

    # Time when we think that the event started
    originating_event_time = mc[0].date

    # Accumulate the data in space and time to increase the signal to noise
    # ratio
    print(' - Performing spatial summing of HPC data.')
    mc = mapcube_tools.accumulate(mapcube_tools.superpixel(mc, spatial_summing),
                                  accum)

    # Might want to do the next couple of steps at this level of the code, since
    # the mapcube could get large.
    # Unravel the data
    print(' - Performing HPC to HG unraveling.')
    unraveled = util.map_unravel(mc,
                                 unraveling_hpc2hg_parameters,
                                 verbose=False)

    # Superpixel values must divide into dimensions of the map exactly.  The
    # oversampling above combined with the superpixeling reduces the explicit
    # effect of
    superpixel = (along_wavefront_sampling, perpendicular_to_wavefront_sampling)*u.pixel
    if np.mod(unraveled[0].dimensions.x.value, superpixel[0].value) != 0:
        raise ValueError('Superpixel values must divide into dimensions of the map exactly: x direction')
    if np.mod(unraveled[0].dimensions.y.value, superpixel[1].value) != 0:
        raise ValueError('Superpixel values must divide into dimensions of the map exactly: y direction')

    print(' - Performing HG superpixel summing.')
    processed = []
    for m in unraveled:
        processed.append(m.superpixel(superpixel))

    # AWARE image processing
    print(' - Performing AWARE processing.')
    umc = aware.processing(Map(processed, cube=True),
                           radii=radii,
                           histogram_clip=[0.0, 99.])

    # Get and store the dynamics of the wave front
    # Note that the error in the position of the wavefront (when measured as
    # the maximum should also include the fact that the wavefront maximum can
    # be anywhere inside that pixel
    results.append(aware.dynamics(umc,
                                  originating_event_time=originating_event_time,
                                  error_choice=error_choice,
                                  position_choice=position_choice,
                                  returned=['answer', 'arc'],
                                  ransac_kwargs=None,
                                  n_degree=1))

    """
    results.append(aware.dynamics(umc,
                                  originating_event_time=originating_event_time,
                                  error_choice=error_choice,
                                  position_choice=position_choice,
                                  returned=['answer'],
                                  ransac_kwargs=ransac_kwargs,
                                  n_degree=2))
    """

#
# Save the results
#
if not os.path.exists(otypes_dir['pkl']):
    os.makedirs(otypes_dir['pkl'])
filepath = os.path.join(otypes_dir['pkl'], otypes_filename['pkl'] + '.pkl')
print('Results saved to %s' % filepath)
f = open(filepath, 'wb')
pickle.dump(results, f)
f.close()

