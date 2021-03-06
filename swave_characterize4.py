#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.units as u
import map_hpc_hg_transforms
from sunpy.map import Map
from sunpy.time import parse_time
from astropy.visualization import LinearStretch
from astropy.visualization.mpl_normalize import ImageNormalize

# Main AWARE processing and detection code
import aware5

# Extra utilities for AWARE
import aware_utils

# AWARE map and mapcube transform utilities
from map_hpc_hg_transforms import MapCubeCoordinateFrameRotatedToNewNorth

# Mapcube handling tools
import mapcube_tools

# Code to create a simulated wave
from sim.wave2d import wave2d

# Parameters for the simulated wave
import swave_params

# Details of the study
import swave_study as sws


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
griddata_method = sws.griddata_methods


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

# Is this run for the AWARE paper?
for_paper = sws.for_paper

# Test the symmetry of the wave - only useful if the wave is actually meant to
# be symmetrical about (0,0)
test_symmetry_of_wave = True

###############################################################################
###############################################################################
#
# Everything below here is set from above
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
                aware_utils.convert_dict_to_single_string(ransac_kwargs)]:
        idir = os.path.join(idir, loc)
        filename = filename + loc + '.'
    filename = filename[0: -1]
    if not(os.path.exists(idir)):
        os.makedirs(idir)
    otypes_dir[ot] = idir
    otypes_filename[ot] = filename

# where to save images
img_filepath = os.path.join(otypes_dir['img'], otypes_filename['img'])
develop = {'img': os.path.join(otypes_dir['img'], otypes_filename['img']),
           'dat': os.path.join(otypes_dir['dat'], otypes_filename['dat'])}

# wave progress map
c_map_cm = cm.nipy_spectral

# Go through all the test waves, and apply AWARE.
results = []
for i in range(0, n_random):
    # Let the user which trial is happening
    print(' - wave name = %s' % wave_name)
    print(' - special designation = %s' % special_designation)
    print(' - position choice = %s' % position_choice)
    print(' - error choice = %s' % error_choice)
    print(' - along wavefront sampling = %i' % along_wavefront_sampling)
    print(' - perpendicular to wavefront sampling = %i' % perpendicular_to_wavefront_sampling)
    print(' - RANSAC parameters = %s' % str(ransac_kwargs))
    print(' - starting trial %i out of %i\n' % (i + 1, n_random))

    if not observational:
        print('\nSimulating %s ' % wave_name)
        if not use_saved:
            # Simulate the wave and return a dictionary
            print(" - Creating test waves.")

            # Load in the wave params
            simulated_wave_parameters = swave_params.waves()[wave_name]

            # Simulate the waves
            euv_wave_data = wave2d.simulate(simulated_wave_parameters,
                                            max_steps, verbose=True,
                                            output=['finalmaps','raw', 'transformed', 'noise'],
                                            use_transform2=use_transform2)

            # Epicenter of the wave front
            epi_lon = simulated_wave_parameters['epi_lon']
            epi_lat = simulated_wave_parameters['epi_lat']

            # Test the wave
            if test_symmetry_of_wave:
                aware_utils.test_symmetry_of_wave(euv_wave_data['transformed'])

            if save_test_waves:
                print(" - Saving test waves.")
                file_path = os.path.join(otypes_dir['dat'], otypes_filename['dat'] + '.pkl')
                print('Saving to %s' % file_path)
                f = open(file_path, 'wb')
                pickle.dump(euv_wave_data, f)
                f.close()
        else:
            print(" - Loading test waves.")
            file_path = os.path.join(otypes_dir['dat'], otypes_filename['dat'] + '.pkl')
            print('Loading from %s' % file_path)
            f = open(file_path, 'rb')
            out = pickle.load(f)
            f.close()
    else:
        # Load observational data from file
        euv_wave_data = aware_utils.create_input_to_aware_for_test_observational_data(wave_name)

    # Storage for the results from all methods and polynomial fits
    print(' - Using the griddata method %s.' % griddata_method)

    # Which data to use
    print('Using the %s data source' % analysis_data_sources)

    # Get the final map out from the wave simulation
    mc = euv_wave_data[analysis_data_sources]

    # Transform back to HG
    #transform_hpc2hg_parameters['epi_lat'] = 0.0 * u.degree
    #transform_hpc2hg_parameters['epi_lon'] = 0.0 * u.degree
    #hg = map_hpc_hg_transforms.mapcube_hpc_to_hg(euv_wave_data["transformed"], transform_hpc2hg_parameters)

    # Accumulate the data in space and time to increase the signal
    # to noise ratio
    print(' - Performing spatial summing of HPC data.')
    mc = mapcube_tools.accumulate(mapcube_tools.superpixel(mc, spatial_summing), temporal_summing)

    if develop is not None:
        aware_utils.write_movie(mc, img_filepath + '_accummulated_data')
    # Swing the position of the start of the longitudinal
    # unwrapping
    for ils, longitude_start in enumerate(longitude_starts):

        # Which angle to start the longitudinal unwrapping
        transform_hpc2hg_parameters['longitude_start'] = longitude_start

        # Which version of AWARE to use
        if aware_version == 0:
            #
            # AWARE version 0 - first do the image processing
            # to isolate the wave front, then do the transformation into
            # heliographic co-ordinates to measure the wavefront.
            #
            print(' - Performing AWARE v0 image processing.')
            print('   - input images are in HPC co-ordinates.')
            aware_processed = aware5.processing(mc,
                                                                   develop=None,
                                                                   radii=radii,
                                                                   func=intensity_scaling_function,
                                                                   histogram_clip=histogram_clip)
            print(' - Segmenting the data to get the emission due to wavefront')
            print('   - calculating wave progress mask.')
            wave_progress_mask = aware_utils.progress_mask(aware_processed)

            print('   - calculating running different persistance images.')
            rdpi = mapcube_tools.running_difference(mapcube_tools.persistence(mc))

            print('   - calculating segmented maps')
            segmented_maps = mapcube_tools.multiply(wave_progress_mask, rdpi)

            print(' - Calculating times of each segmented map')
            times = [m.date for m in segmented_maps]

        elif aware_version == 1:
            """
            #
            # AWARE version 1 - first transform in to the heliographic co-ordinates
            # then do the image processing.
            #
            print(' - Performing AWARE v1 image processing.')
            print(' - Performing HPC to HG unraveling.')
            unraveled = mapcube_hpc_to_hg(mc,
                                          transform_hpc2hg_parameters,
                                          verbose=False,
                                          method=griddata_method)

            # Superpixel values must divide into dimensions of the
            # map exactly. The oversampling above combined with the
            # superpixeling reduces the explicit effect of
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
            umc = aware5.processing(Map(processed, cube=True),
                                    radii=radii,
                                    func=intensity_scaling_function,
                                    histogram_clip=histogram_clip)
            """
        # Get the number of longitude and latitude edges that make up the bins
        nlon_edges = np.int(transform_hpc2hg_parameters['lon_num'].value)
        nlat_edges = np.int(transform_hpc2hg_parameters['lat_num'].value)

        # Extract the data from the AWARE processed maps
        print(' - rotating AWARE processed data to new north')
        rotated_to_wave_source = MapCubeCoordinateFrameRotatedToNewNorth(segmented_maps, epi_lon, epi_lat)
        print(' - extracting data')
        hgnn_data, lon_bins, lat_bins = rotated_to_wave_source.extract(nlon_edges=nlon_edges, nlat_edges=nlat_edges)
        print(' - removing non-finite data')
        hgnn_data[~np.isfinite(hgnn_data)] = 0.0
        print(' - flip data.')
        hgnn_data = hgnn_data[:, ::-1, :]

        # Longitude
        longitude = np.mean(lon_bins, axis=0)
        lon_bin = longitude[1] - longitude[0]

        # Latitudinal propagation distance
        latitude = np.mean(lat_bins, axis=0)
        latitude = np.abs(latitude - latitude[0])
        lat_bin = latitude[1] - latitude[0]

        # Define the mapcube that will be used to define the
        # location of the wavefront.
        # Options...
        # 1. just use the result of AWARE image processing
        # 2. Multiple the AWARE progress map with the RDP to get the
        # location of the wavefront.

        # Fit the arcs
        print(' - Fitting polynomials to arcs')
        longitude_fit = []
        nlon = hgnn_data.shape[0]
        for lon in range(0, nlon-1):
            # Get the next arc
            arc = aware5.Arc(hgnn_data[lon, :, :], times,
                             latitude, longitude[lon],
                             start_time=mc[0].date,
                             sigma=np.sqrt(hgnn_data[lon, :, :]))

            # Measure the location of the arc and estimate an
            # error in the location
            position, position_error = arc.locator(position_choice, error_choice)
            # Get the dynamics of the arcs
            polynomial_degree_fit = []
            for n_degree in n_degrees:
                analysis = Analysis()
                analysis.aware_version = aware_version
                analysis.method = griddata_method
                analysis.n_degree = n_degree
                analysis.lon = lon
                analysis.ils = ils
                analysis.answer = aware5.FitPosition(arc.t,
                                                     position,
                                                     position_error,
                                                     ransac_kwargs=ransac_kwargs,
                                                     fit_method=fit_method,
                                                     n_degree=n_degree,
                                                     arc_identity=arc.longitude,
                                                     error_tolerance_kwargs=error_tolerance_kwargs)
                # Store each polynomial degree
                polynomial_degree_fit.append(analysis)
            # Store the fits at this longitude
            longitude_fit.append(polynomial_degree_fit)
        # results are stored as results[longitude_index][n=1 polynomial, n=2 polynomial]
        results.append(longitude_fit)


#
# Save the fit results
#
if not os.path.exists(otypes_dir['dat']):
    os.makedirs(otypes_dir['dat'])
filepath = os.path.join(otypes_dir['dat'], otypes_filename['dat'] + '.pkl')
print('Results saved to %s' % filepath)
f = open(filepath, 'wb')
pickle.dump(results, f)
f.close()


#
#
#
if not observational:
    speed = simulated_wave_parameters['speed'][0]
    acceleration = simulated_wave_parameters['acceleration']
    d0 = parse_time(euv_wave_data['finalmaps'][0].date)
    time = np.asarray([(parse_time(m.date) - d0).total_seconds() for m in euv_wave_data['finalmaps']]) * u.s
    true_position = speed * time + 0.5 * acceleration * time * time
    line = {"t": time, "y": true_position, "kwargs": {"label": "true position"}}

#
# Invert the AWARE detection cube back to helioprojective Cartesian
#
if aware_version == 1:
    pass
    """
    transform_hg2hpc_parameters = {'epi_lon': transform_hpc2hg_parameters['epi_lon'],
                                   'epi_lat': transform_hpc2hg_parameters['epi_lat'],
                                   'xnum': 1024*u.pixel,
                                   'ynum': 1024*u.pixel}

    # Transmogrify
    umc_hpc = mapcube_hg_to_hpc(umc, transform_hg2hpc_parameters, method=griddata_method)

    # Save the wave results
    if not os.path.exists(otypes_dir['dat']):
        os.makedirs(otypes_dir['dat'])
    if not os.path.exists(otypes_dir['img']):
        os.makedirs(otypes_dir['img'])
    filepath = os.path.join(otypes_dir['dat'], otypes_filename['dat'] + '.wave_hpc.dat')
    print('Results saved to %s' % filepath)
    f = open(filepath, 'wb')
    pickle.dump(umc_hpc, f)
    f.close()

    # Create the wave progress map
    wave_progress_map, timestamps = aware_utils.progress_map(umc_hpc)
    angle = 180*u.deg
    use_disk_mask = True
    """
else:
    wave_progress_map, timestamps = aware_utils.progress_map(aware_processed)
    angle = 0*u.deg
    use_disk_mask = False


# Find the on-disk locations
disk = np.zeros_like(wave_progress_map.data)
nx = disk.shape[1]
ny = disk.shape[0]
cx = wave_progress_map.center.x.to(u.arcsec).value
cy = wave_progress_map.center.y.to(u.arcsec).value
r = wave_progress_map.rsun_obs.to(u.arcsec).value
xloc = np.zeros(nx)
for i in range(0, nx-1):
    xloc[i] = cx - wave_progress_map.pixel_to_data(i * u.pix, 0*u.pix)[0].to(u.arcsec).value

yloc = np.zeros(ny)
for j in range(0, ny-1):
    yloc[j] = cy - wave_progress_map.pixel_to_data(0*u.pix, j*u.pix)[1].to(u.arcsec).value

for i in range(0, nx-1):
    for j in range(0, ny-1):
        disk[i, j] = np.sqrt(xloc[i]**2 + yloc[j]**2) < r

# Zero out the off-disk locations
if use_disk_mask:
    wpm_data = wave_progress_map.data * disk
else:
    wpm_data = wave_progress_map.data
wp_map = Map(wpm_data, wave_progress_map.meta).rotate(angle=angle)
wp_map.plot_settings['norm'] = ImageNormalize(vmin=1, vmax=len(timestamps), stretch=LinearStretch())
wp_map.plot_settings['cmap'] = c_map_cm
wp_map.plot_settings['norm'].vmin = 1

# Create a wave progress map.  This is a composite map with a colorbar that
# shows timestamps corresponding to the progress of the wave, and a typical
# image from the time of the wave.
# TODO: make the zero value pixels completely transparent

# Observation date
observation_date = mc[0].date.strftime("%Y-%m-%d")

# Create the composite map
c_map = Map(mc[0], wp_map, composite=True)

# Create the figure
plt.close('all')
figure = plt.figure()
axes = figure.add_subplot(111)
if for_paper:
    observation = r"AIA {:s}".format(mc[0].measurement._repr_latex_())
    title = "wave progress map\n{:s}".format(observation)
    image_file_type = 'eps'
else:
    title = "{:s} ({:s})".format(observation_date, wave_name)
    image_file_type = 'png'
ret = c_map.plot(axes=axes, title=title)
c_map.draw_limb()
c_map.draw_grid()

# Set up the color bar
nticks = 6
timestamps_index = np.linspace(1, len(timestamps)-1, nticks, dtype=np.int).tolist()
cbar_tick_labels = []
for index in timestamps_index:
    wpm_time = timestamps[index].strftime("%H:%M:%S")
    cbar_tick_labels.append(wpm_time)
cbar = figure.colorbar(ret[1], ticks=timestamps_index)
cbar.ax.set_yticklabels(cbar_tick_labels)
cbar.set_label('time (UT) ({:s})'.format(observation_date))
cbar.set_clim(vmin=1, vmax=len(timestamps))

# Save the wave progress map
plt.savefig(img_filepath + '_wave_progress_map.{:s}'.format(image_file_type))


# Write movie of wave progress across the disk
plt.close('all')

"""
def draw_limb(fig, ax, sunpy_map):
    p = sunpy_map.draw_limb()
    return p
pm = aware_utils.progress_mask(aware_processed)
new_pm = []
for im, m in enumerate(pm):
    new_data = pm[im].data * (1+im)
    new_map = Map(new_data, pm.meta)
    new_map.plot_settings['cmap'] = c_map_cm
    new_map.plot_settings['norm'] = ImageNormalize(vmin=0, vmax=len(timestamps), stretch=LinearStretch())
    new_pm.append(new_map)

aware_utils.write_movie(Map(new_pm, cube=True), img_filepath + '_aware_processed')
"""

