#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from sunpy.map import Map
plt.ion()

# Main AWARE processing and detection code
import aware

# AWARE map and mapcube transform utilities
from map_hpc_hg_transforms import mapcube_hpc_to_hg, mapcube_hg_to_hpc

# Mapcube handling tools
import mapcube_tools

# Code to create a simulated wave
from sim.wave2d import wave2d

# Parameters for the simulated wave
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
example = 'wavenorm4_slow'
#example = 'no_noise_no_solar_rotation_slow_360'

# Use pre-saved data
use_saved = False

# Number of trials
ntrials = 100

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
params = swave_params.waves(lon_start=-180 * u.degree + 0 * u.degree)[example]

# Unraveling parameters used to convert HPC image data to HG data.  Trying
# oversampling in order to get a good sampling on the wavefront.  Then use
# superpixels of this transformed data to get the final HG image we will use to
# do the wave detection.
transform_hpc2hg_parameters = {'lon_bin': 1.0*u.degree,
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

    transform_hg2hpc_parameters = {'epi_lon': params['epi_lon'],
                                   'epi_lat': params['epi_lat'],
                                   'xnum': 800*u.pixel,
                                   'ynum': 800*u.pixel}

    # Storage for the results from all methods and polynomial fits
    final = {}
    for method in griddata_methods:
        print(' - Using the griddata method %s.' % method)
        final[method] = []

        # Which data to use
        for source in analysis_data_sources:
            print('Using the %s data source' % source)
            if source == 'finalmaps':
                # Get the final map out from the wave simulation
                mc = out['finalmaps']

                # Accumulate the data in space and time to increase the signal
                # to noise ratio
                print(' - Performing spatial summing of HPC data.')
                mc = mapcube_tools.accumulate(mapcube_tools.superpixel(mc, spatial_summing), accum)

                # Might want to do the next couple of steps at this level of the
                # code, since the mapcube could get large. Unravel the data
                print(' - Performing HPC to HG unraveling.')
                unraveled = mapcube_hpc_to_hg(mc, transform_hpc2hg_parameters,
                                                  verbose=False,
                                                  method=method)
            if source == 'raw':
                # Use the raw HG maps and apply the accumulation in the space
                # and time directions.
                mc = mapcube_hg_to_hpc(out['raw'], transform_hg2hpc_parameters)
                mc = mapcube_tools.accumulate(mapcube_tools.superpixel(mc, spatial_summing), accum)
                unraveled = mapcube_hpc_to_hg(mc, transform_hpc2hg_parameters)

            if source == 'raw_no_accumulation':
                # Use the raw HG maps with NO accumulation in the space and
                # time directions.
                unraveled = out['raw']

            # Superpixel values must divide into dimensions of the map exactly.
            # The oversampling above combined with the superpixeling reduces the
            # explicit effect of
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
            print(' - Performing AWARE image processing.')
            umc = aware.processing(Map(processed, cube=True),
                                   radii=radii,
                                   histogram_clip=[0.0, 99.])

            # Get the data out
            umc_data = umc.as_array()

            # Longitude
            lon_bin = umc[0].scale[0]  # .to('degree/pixel').value
            nlon = np.int(umc[0].dimensions[0].value)
            longitude = np.min(umc[0].xrange) + np.arange(0, nlon) * u.pix * lon_bin

            # Latitude
            lat_bin = umc[0].scale[1]  # .to('degree/pixel').value
            nlat = np.int(umc[0].dimensions[1].value)
            latitude = np.min(umc[0].yrange) + np.arange(0, nlat) * u.pix * lat_bin

            # Times
            times = aware.get_times_from_start(umc, start_date=mc[0].date)

            for lon in range(0, nlon):
                # Get the next arc
                arc = aware.Arc(umc_data[:, lon, :], times, latitude, longitude[lon])

                # Convert the arc information into data that we can use to fit
                arc_as_fit = aware.arc_as_fit(arc, error_choice=error_choice, position_choice=position_choice)

                # Get the dynamics of the arcs
                polynomial_degree_fit = []
                for n_degree in (1, 2):
                    polynomial_degree_fit.append(aware.dynamic(arc_as_fit[0],
                                                               arc_as_fit[1],
                                                               arc_as_fit[2],
                                                               ransac_kwargs=None,
                                                               n_degree=n_degree))
                final[method].append(polynomial_degree_fit)


    # Store the results from all the griddata methods and polynomial fits
    results.append(final)

    # Simple summary of the fit.
    # v = [x.velocity.value if x.fit_able else np.nan for x in dynamics]


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



"""
AWARE - a description of  basic algorithm

Version 0
---------
Version 0 is the first version of the AWARE algorithm as originally developed.

1. Get HPC image data.
2. Accumulate in the space direction to increase SNR.
3. Accumulate in the time direction to increase SNR.
4. Create a mapcube of the data.
5. Calculate the persistence transform of the mapcube.
6. Calculate the running difference of the mapcube.
7. Apply a noise reduction filter (for example, median filter) on multiple
   length-scales.
8. Apply an operation to rejoin broken parts of the wavefront (for example, a
   morphological closing).
9. Transform the mapcube into the HG co-ordinate system based on a center that
   is the point where the wave originates from.
10. Measure the progress of the wavefront along arcs.


Version 1
---------
Version 1 is the version of the AWARE algorithm implemented here.  The order of
the above steps is slightly different, and the choices listed as "examples"
have been used.  The order of the Version 1 algorithm is as listed below

1, 2, 3, 4, 9, 5, 6, 7, 8, 10

The idea behind this is that the noise cleaning and morphological operations
should be used in the HG co-ordinate system where the effect of the Sun's
curvature has been removed (or at least compensated for.)


Ideas
-----

(A)
Fit models to arcs generated using Version 1 with griddata set to 'linear' and
'nearest'.
 - average the results

Fit models to arcs generated using Version 0 with griddata set to 'linear' and
'nearest'.
 - average the results




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