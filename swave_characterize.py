#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import aware_utils

#
plt.ion()

#
from sunpy.map import Map

# Main AWARE processing and detection code
import aware

# AWARE utilities
import util

# Simulated wave parameters
from sim.wave2d import wave2d

# Mapcube handling tools
import mapcube_tools

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
random_seed = 42
np.random.seed(random_seed)

# Select the wave
#example = 'wavenorm4_slow'
example = 'no_noise_no_solar_rotation_slow_360'

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
along_wavefront_sampling = 5

# Oversampling perpendicular to wavefront
perpendicular_to_wavefront_sampling = 5

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

# Analysis source data
sources = ('finalmaps', 'raw', 'raw_no_processing')
sources = ('finalmaps',)

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
        out = wave2d.simulate(params,
                              max_steps,
                              verbose=True,
                              output=['raw', 'transformed', 'noise', 'finalmaps'],
                              use_transform2=True)
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

    reraveling_hg2hpc_parameters = {'epi_lon': params['epi_lon'],
                                    'epi_lat': params['epi_lat'],
                                    'xnum': 800*u.pixel,
                                    'ynum': 800*u.pixel}

    for source in sources:
        print('Using the %s source' % source)
        if source == 'finalmaps':
            # Get the final map out
            mc = out['finalmaps']

            # Accumulate the data in space and time to increase the signal to noise
            # ratio
            print(' - Performing spatial summing of HPC data.')
            mc = mapcube_tools.accumulate(mapcube_tools.superpixel(mc, spatial_summing),
                                          accum)

            # Might want to do the next couple of steps at this level of the code, since
            # the mapcube could get large.
            # Unravel the data
            print(' - Performing HPC to HG unraveling.')
            unraveled = util.mapcube_unravel(mc,
                                             unraveling_hpc2hg_parameters,
                                             verbose=False,
                                             method='linear')
        if source == 'raw':
            mc = util.mapcube_reravel(out['raw'], reraveling_hg2hpc_parameters)
            mc = mapcube_tools.accumulate(mapcube_tools.superpixel(mc, spatial_summing),
                                          accum)
            unraveled = util.mapcube_unravel(mc, unraveling_hpc2hg_parameters)

        if source == 'raw_no_processing':
            unraveled = out['raw']

        # Time when we think that the event started
        originating_event_time = mc[0].date

        # Superpixel values must divide into dimensions of the map exactly.  The
        # oversampling above combined with the superpixeling reduces the explicit
        # effect of
        hg_superpixel = (along_wavefront_sampling, perpendicular_to_wavefront_sampling)*u.pixel
        if np.mod(unraveled[0].dimensions.x.value, hg_superpixel[0].value) != 0:
            raise ValueError('Superpixel values must divide into dimensions of the map exactly: x direction')
        if np.mod(unraveled[0].dimensions.y.value, hg_superpixel[1].value) != 0:
            raise ValueError('Superpixel values must divide into dimensions of the map exactly: y direction')

        print(' - Performing HG superpixel summing.')
        processed = []
        for m in unraveled:
            processed.append(m.superpixel(hg_superpixel))

        # AWARE image processing
        print(' - Performing AWARE processing.')
        umc = aware.processing(Map(processed, cube=True),
                               radii=radii,
                               histogram_clip=[0.0, 99.])

        # Get all the arcs
        arcs = aware.get_arcs(umc, originating_event_time=originating_event_time)

        # Convert the arc information into data that we can use to fit
        arcs_as_fit = aware.arcs_as_fit(arcs,
                                        error_choice=error_choice,
                                        position_choice=position_choice)

        # Get the dynamics of the arcs
        dynamics = aware.dynamics(arcs_as_fit,
                                  ransac_kwargs=None,
                                  n_degree=1)

        # Simple summary of the fit.
        v = [x.velocity.value if x.fit_able else np.nan for x in dynamics]


"""
for source in ('finalmaps', 'raw', 'raw_no_processing'):
    plt.plot(v[source], label=source)
initial_speed = (params['speed'][0] * aware_utils.solar_circumference_per_degree).to('km/s').value
plt.axhline(initial_speed, label='true velocity')
plt.legend()


stop
#
# Testing the util versions of ravel and unravel
#
hg2hpc = util.map_reravel(out['raw'], reraveling_hg2hpc_parameters)
hg2hpc2hg = util.map_unravel(hg2hpc, unraveling_hpc2hg_parameters)

results2 = []
results2.append(aware.dynamics(hg2hpc2hg,
                                  originating_event_time=originating_event_time,
                                  error_choice=error_choice,
                                  position_choice=position_choice,
                                  returned=['answer'],
                                  ransac_kwargs=ransac_kwargs,
                                  n_degree=2))

v2 = [x[0].velocity.value if x[0].fit_able else np.nan for x in results2[:]]

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

"""

"""
Notes:

Using 'no_noise_no_solar_rotation_slow_360'

Even using the raw output with this data results in velocities higher than the
true velocity.  This is because at earlier times the wave is not in the field
of view, and so its location is incorrectly reported.

Doing the round trip HG -> HPC -> HG starting with the raw data,
introduces the variation across the wavefront, and the decreased velocity.
The error that is introduced is small compared to the velocity, and likely
much smaller than other sources of error.  Investigating possible mitigation
strategies involving oversampling.


"""