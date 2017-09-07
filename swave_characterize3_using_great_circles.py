#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
# Requires SunPy 0.7
#
import os
import pickle
from copy import deepcopy

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from matplotlib.colors import Normalize

import astropy.units as u
from astropy.visualization import LinearStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.coordinates import SkyCoord

from sunpy.map import Map
from sunpy.time import parse_time
import sunpy.coordinates

# Main AWARE processing and detection code
import aware5_without_swapping_emission_axis

# Extra utilities for AWARE
import aware_utils

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

# Is this run for the AWARE paper?
for_paper = sws.for_paper

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
    otypes_filename[ot] = filename + '.' + str(great_circle_points)

# where to save images
img_filepath = os.path.join(otypes_dir['img'], otypes_filename['img'])
develop = {'img': os.path.join(otypes_dir['img'], otypes_filename['img']),
           'dat': os.path.join(otypes_dir['dat'], otypes_filename['dat'])}

# Answer storage main list
results = []

# Go through all the test waves, and apply AWARE.
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

            # Transform parameters used to convert HPC image data to HG data.
            # The HPC data is transformed to HG using the location below as the
            # "pole" around which the data is transformed
            transform_hpc2hg_parameters['epi_lon'] = -simulated_wave_parameters['epi_lon']
            transform_hpc2hg_parameters['epi_lat'] = -simulated_wave_parameters['epi_lat']

            # Simulate the waves
            euv_wave_data = wave2d.simulate(simulated_wave_parameters,
                                            max_steps, verbose=True,
                                            output=['finalmaps','raw', 'transformed', 'noise'],
                                            use_transform2=use_transform2)
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

        # Which data to use
        print('Using the %s data source' % analysis_data_sources)

        # Get the final map out from the wave simulation
        hpc_maps = euv_wave_data[analysis_data_sources]
    else:
        # Load observational data from file
        print('Loading observational data - {:s}'.format(wave_name))
        euv_wave_data = aware_utils.create_input_to_aware_for_test_observational_data(wave_name)
        hpc_maps = euv_wave_data['finalmaps']

        # Transform parameters used to convert HPC image data to HG data.
        # The HPC data is transformed to HG using the location below as the
        # "pole" around which the data is transformed
        transform_hpc2hg_parameters['epi_lon'] = euv_wave_data['epi_lon'] * u.deg
        transform_hpc2hg_parameters['epi_lat'] = euv_wave_data['epi_lat'] * u.deg

    # Storage for the results from all methods and polynomial fits
    print(' - Using the griddata method %s.' % griddata_method)

    ############################################################################
    # Accumulate the data in space and time to increase the signal
    # to noise ratio
    print(' - Performing spatial summing of HPC data.')
    mc = mapcube_tools.accumulate(mapcube_tools.superpixel(hpc_maps, spatial_summing), temporal_summing)
    #if develop is not None:
    #    aware_utils.write_movie(mc, img_filepath + '_accummulated_data')

    ############################################################################
    # Initial map that shows an image of the Sun.  First data used in all
    # further analysis
    initial_map = deepcopy(mc[0])

    ############################################################################
    # Define the estimated initiation point
    initiation_point = SkyCoord(transform_hpc2hg_parameters['epi_lon'],
                                transform_hpc2hg_parameters['epi_lat'],
                                frame='heliographic_stonyhurst').transform_to(initial_map.coordinate_frame)

    # Swing the position of the start of the longitudinal
    # unwrapping
    for ils, longitude_start in enumerate(longitude_starts):

        # Which angle to start the longitudinal unwrapping
        transform_hpc2hg_parameters['longitude_start'] = longitude_start

        #
        # AWARE version 0 - first do the image processing
        # to isolate the wave front, then do the transformation into
        # heliographic co-ordinates to measure the wavefront.
        #
        print(' - Performing AWARE v0 image processing.')
        aware_processed = aware5_without_swapping_emission_axis.processing(mc,
                                            develop=None,
                                            radii=radii,
                                            func=intensity_scaling_function,
                                            histogram_clip=histogram_clip)
        print(' - Segmenting the data to get the emission due to wavefront')
        segmented_maps = mapcube_tools.multiply(aware_utils.progress_mask(aware_processed),
                                                mapcube_tools.running_difference(mapcube_tools.persistence(mc)))

        # Times
        times = [m.date for m in segmented_maps]

        # Map for locations that participate in the fit
        fit_participation_datacube = np.zeros_like(segmented_maps.as_array())

        # Calculate the arcs
        print(' - Creating arc location information')
        # Number of arcs
        nlon = 360

        # Number of files that we are examining
        nt = len(segmented_maps)

        # Equally spaced arcs
        angles = (np.linspace(0, 2*np.pi, 361))[0:-1] * u.rad

        # Calculate co-ordinates in a small circle around the launch point
        r = 1*u.arcsec
        x = r*np.sin(angles)
        y = r*np.cos(angles)
        locally_circular = SkyCoord(initiation_point.Tx + x,
                                    initiation_point.Ty + y,
                                    frame=initial_map.coordinate_frame)

        # Calculate all the arcs a
        extract = []
        for lon in range(0, nlon):
            # Calculate the great circle
            great_circle = aware_utils.GreatCircle(initiation_point,
                                                   locally_circular[lon],
                                                   points=great_circle_points)

            # Get the coordinates of the great circle
            coordinates = great_circle.coordinates()

            # Get the arc from the start to limb
            arc_from_start_to_back = coordinates[0:great_circle.from_front_to_back_index]

            # Calculate which pixels the extract from the map
            integer_pixels = np.asarray(np.rint(arc_from_start_to_back.to_pixel(initial_map.wcs)), dtype=int)

            """
            # Get where these pixels are on the Sun in co-ordinates
            integer_pixel_coordinates = initial_map.pixel_to_data(integer_pixels[0]*u.pix, integer_pixels[1]*u.pix)

            # Convert to SkyCoords
            integer_pixel_skycoords = SkyCoord(integer_pixel_coordinates[0], integer_pixel_coordinates[1], frame=initial_map.coordinate_frame)

            # Calculate the inner angles
            integer_pixel_skycoords_inner_angle = np.zeros(shape=len(integer_pixel_skycoords))
            for ips in range(0, len(integer_pixel_skycoords)):
                integer_pixel_skycoords_inner_angle[ips] = aware_utils.InnerAngle(initiation_point,
                                                                      integer_pixel_skycoords[ips]).inner_angle.value

            # Get the latitude
            latitude = (integer_pixel_skycoords_inner_angle * u.rad).to(u.deg)
            """
            # Latitudinal extent.  Note that the inner angles are not quite correct for
            # the pixels used.  This is because the pixel values used to extract the data are
            # integer values, whereas the pixel values returned are non-integer and the
            # corresponding inner angles refer to these non-integer pixel values.  This is fixed in
            # the commented-out code above
            inner_angles = great_circle.inner_angles()
            latitude = inner_angles[0:great_circle.from_front_to_back_index].to(u.deg).flatten()

            # Store the results
            extract.append((integer_pixels, latitude, arc_from_start_to_back))

        # Fit the arcs
        print(' - Fitting polynomials to arcs')
        longitude_fit = []
        for lon in range(0, nlon):
            # At each longitude perform a number of fits as required.
            lat_time_data = aware5_without_swapping_emission_axis.build_lat_time_data(lon, extract, segmented_maps)

            # Define the next arc
            pixels = extract[lon][0]
            latitude = extract[lon][1]
            arc = aware5_without_swapping_emission_axis.Arc(lat_time_data, times, latitude, angles[lon].to(u.deg),
                             start_time=initial_map.date, sigma=np.sqrt(lat_time_data))
            # Measure the location of the wave and estimate an
            # error in its location
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
                analysis.answer = aware5_without_swapping_emission_axis.FitPosition(arc.t,
                                                     position,
                                                     position_error,
                                                     ransac_kwargs=ransac_kwargs,
                                                     fit_method=fit_method,
                                                     n_degree=n_degree,
                                                     arc_identity=arc.longitude,
                                                     error_tolerance_kwargs=error_tolerance_kwargs)
                # Store a (lat, lon, time) cube that indicates where a fit was
                # made
                # Store each polynomial degree
                polynomial_degree_fit.append(analysis)

                # Update the fit participation mask
                if analysis.answer.fitted:
                    time_indices_fitted = analysis.answer.indicesf
                    for k in range(0, len(time_indices_fitted)):
                        x = pixels[0, :]
                        y = pixels[1, :]
                        fit_participation_datacube[y[:], x[:], time_indices_fitted[k]] = 1

            # Store the fits at this longitude
            longitude_fit.append(polynomial_degree_fit)
        # results are stored as results[longitude_index][n=1 polynomial,
        # n=2 polynomial]
        results.append(longitude_fit)

################################################################################
# color choices
# map color choices
base_cm_sun_image = cm.gray_r
base_cm_wave_progress = cm.plasma
base_cm_long_score = cm.viridis
fitted_arcs_progress_map_cm = cm.plasma

# Limb formatting
draw_limb_kwargs = {"color": "c"}

# Grid formatting
draw_grid_kwargs = {"color": "c"}

# indication of the epicenter
epicenter_kwargs = {"edgecolor": 'w', "facecolor": "c", "radius": 50,
                    "fill": True, "zorder": 1000}

# Guide lines on the sphere
line = {0: {"kwargs": {"linestyle": "solid", "color": "k", "linewidth": 1.0, "zorder": 1003}},
        90: {"kwargs": {"linestyle": "dashed", "color": "k", "linewidth": 1.0, "zorder": 1003}},
        180: {"kwargs": {"linestyle": "dashdot", "color": "k", "linewidth": 1.0, "zorder": 1003}},
        270: {"kwargs": {"linestyle": "dotted", "color": "k", "linewidth": 1.0, "zorder": 1003}}}

# Long score formatting
bls_kwargs = {"color": "r", "zorder": 1001, "linewidth": 2}
fitted_arc_kwargs = {"linewidth": 1, "color": 'b'}

# Image of the Sun used as a background
sun_image = deepcopy(initial_map)
sun_image.plot_settings['cmap'] = base_cm_sun_image
observation_date = initial_map.date.strftime("%Y-%m-%d")
observation_datetime = initial_map.date.strftime("%Y-%m-%d %H:%M:%S")
#
image_file_type = 'png'
observation = r"AIA {:s}".format(initial_map.measurement._repr_latex_())

################################################################################
# Save the fit results
#
if not os.path.exists(otypes_dir['dat']):
    os.makedirs(otypes_dir['dat'])
filepath = os.path.join(otypes_dir['dat'], otypes_filename['dat'] + '.pkl')
print('Results saved to %s' % filepath)
f = open(filepath, 'wb')
pickle.dump(results, f)
f.close()


################################################################################
# Create a typical arc line for simulated data
#
if not observational:
    speed = simulated_wave_parameters['speed'][0]
    acceleration = simulated_wave_parameters['acceleration']
    d0 = parse_time(euv_wave_data['finalmaps'][0].date)
    time = np.asarray([(parse_time(m.date) - d0).total_seconds() for m in euv_wave_data['finalmaps']]) * u.s
    true_position = speed * time + 0.5 * acceleration * time * time
    simulated_line = {"t": time, "y": true_position, "kwargs": {"label": "true position"}}


################################################################################
# Create the wave progress map
#
wave_progress_map, timestamps = aware_utils.wave_progress_map_by_location(aware_processed)
wave_progress_map_cm = base_cm_wave_progress
wave_progress_map_cm.set_under(color='w', alpha=0)
wave_progress_map_norm = ImageNormalize(vmin=1, vmax=len(timestamps), stretch=LinearStretch())
wave_progress_map.plot_settings['cmap'] = wave_progress_map_cm
wave_progress_map.plot_settings['norm'] = wave_progress_map_norm

################################################################################
# Create the fit participation mapcube and final map
#
fit_participation_mapcube = []
for i in range(0, nt):
    fit_participation_map = Map(fit_participation_datacube[:, :, i], segmented_maps[i].meta)
    fit_participation_mapcube.append(fit_participation_map)
fit_participation_mapcube = mapcube_tools.multiply(Map(fit_participation_mapcube, cube=True), segmented_maps)
fit_participation_map, _ = aware_utils.wave_progress_map_by_location(fit_participation_mapcube)
fit_participation_map.plot_settings['cmap'] = wave_progress_map_cm
fit_participation_map.plot_settings['norm'] = wave_progress_map_norm

fit_participation_map_mask_data = fit_participation_map.data > 0.0
fit_participation_map_mask = Map(1.0*fit_participation_map_mask_data, fit_participation_map.meta)

###############################################################################
# Create a map of the Long Score
#
# Long score
long_score = np.asarray([aaa[1].answer.long_score.final_score if aaa[1].answer.fitted else 0.0 for aaa in results[0]])

# Best Long score
long_score_argmax = long_score.argmax()

bls_string = (angles[long_score_argmax].to(u.deg))._repr_latex_()

# Make the map data
long_score_map = deepcopy(initial_map)
long_score_map.data[:, :] = -1.0
for lon in range(0, nlon):
    pixels = extract[lon][0]
    x = pixels[0, :]
    y = pixels[1, :]
    if lon == long_score_argmax:
        long_score_value = 200.0
    else:
        long_score_value = long_score[lon]
    long_score_map.data[y[:], x[:]] = long_score_value

# Create the map and set the color map
long_score_map_cm = base_cm_long_score
long_score_map_cm.set_over(color=bls_kwargs["color"], alpha=1.0)
long_score_map_cm.set_under(color='w', alpha=0.0)
long_score_map.plot_settings['cmap'] = long_score_map_cm
long_score_map.plot_settings['norm'] = ImageNormalize(vmin=0, vmax=100, stretch=LinearStretch())
fit_no_participation_index = np.where(fit_participation_map.data == 0.0)
long_score_map.data *= fit_participation_map_mask_data
long_score_map.data[fit_no_participation_index] = -1

###############################################################################
# Find the maximum extent of the best Long score, based on the fit
# participation array.
# long_score_argmax_pixels = extract[long_score_argmax][0]
# x = long_score_argmax_pixels[0, :]
# y = long_score_argmax_pixels[1, :]
# long_score_argmax_pixels_value = fit_participation_map.data[y[:], x[:]]
# long_score_argmax_pixels_nonzero_index = np.nonzero(long_score_argmax_pixels_value)[0][-1]
# long_score_argmax_x = (extract[long_score_argmax][2].Tx.value)[0:long_score_argmax_pixels_nonzero_index]
# long_score_argmax_y = (extract[long_score_argmax][2].Ty.value)[0:long_score_argmax_pixels_nonzero_index]
# long_score_argmax_arc_from_start_to_back = extract[long_score_argmax][2][0:long_score_argmax_pixels_nonzero_index]

bls_answer = results[0][long_score_argmax][1].answer
bls_answer_max_latitudinal_extent = np.max(bls_answer.best_fit[-1])
bls_latitude = (extract[long_score_argmax][1]).value
diff = np.argmin(np.abs(bls_latitude-bls_answer_max_latitudinal_extent))
long_score_argmax_arc_from_start_to_back = extract[long_score_argmax][2][0:diff]


################################################################################
# Find the maximum extent of all the arcs fit and make a map of that

# Mask that will hold the fitted arcs
fitted_arcs_mask = np.zeros_like(long_score_map.data) - 1

# Go through all the longitudes
for lon in range(0, nlon):
    # Next fit
    answer = results[0][lon][1].answer

    # Maximum latitudinal extent
    if answer.fitted:
        answer_max_latitudinal_extent = answer.best_fit[-1]
        answer_min_latitudinal_extent = answer.best_fit[0]

        # Get the latitude of the arc
        latitude = (extract[lon][1]).value

        # Get the pixels along the arc
        pixels = extract[lon][0]

        # Find the index where the arc latitude equals the maximum latitudinal
        # extent of the fit
        max_arg_latitude = np.argmin(np.abs(latitude-answer_max_latitudinal_extent))
        min_arg_latitude = np.argmin(np.abs(latitude-answer_min_latitudinal_extent))

        # Get the x and y pixels of the fitted arc and fill in the arc.
        x = pixels[0, min_arg_latitude:max_arg_latitude]
        y = pixels[1, min_arg_latitude:max_arg_latitude]

        # Calculate the time at all the latitudes
        bfp = answer.estimate
        if answer.n_degree == 2:
            z2 = bfp[1]**2 - 4*bfp[0]*(bfp[2] - latitude[min_arg_latitude:max_arg_latitude])
            fitted_arc_time = (-bfp[1] + np.sqrt(z2))/(2*bfp[0])
        else:
            fitted_arc_time = (latitude[min_arg_latitude:max_arg_latitude] - bfp[1])/bfp[0]
        # Return in units of the summation
        #fitted_arc_time = fitted_arc_time - fitted_arc_time[0] + answer.timef[0]
        fitted_arc_time[fitted_arc_time < 0] = -1
        fitted_arcs_mask[y[:], x[:]] = fitted_arc_time[:]

fitted_arcs_mask[np.isnan(fitted_arcs_mask)] = -1

# Create the fitted arc map
fitted_arcs_progress_map = Map(fitted_arcs_mask, wave_progress_map.meta)
fitted_arcs_progress_map_cm.set_under(color='w', alpha=0)
fitted_arcs_progress_map_norm = ImageNormalize(vmin=0, vmax=np.max(fitted_arcs_progress_map.data), stretch=LinearStretch())
fitted_arcs_progress_map.plot_settings['cmap'] = fitted_arcs_progress_map_cm
fitted_arcs_progress_map.plot_settings['norm'] = fitted_arcs_progress_map_norm


###############################################################################
# Wave progress plot
# Plot and save a composite map with the following features.
# (1) Inverted b/w image of the Sun
# (2) Full on/off disk wave progress map
# (3) Colorbar with timestamps corresponding to the progress of the wave
# (4) Outlined circle showing the location of the putative wave source

# Create the composite map
c_map = Map(sun_image, wave_progress_map, composite=True)

# Create the figure
figure = plt.figure(1)
axes = figure.add_subplot(111)
ret = c_map.plot(axes=axes, title="wave progress map")
c_map.draw_limb(**draw_limb_kwargs)
c_map.draw_grid(**draw_grid_kwargs)

# Add in lines that indicate 0, 90, 180 and 270 degrees
for key in line.keys():
    arc_from_start_to_back = extract[key][2]
    kwargs = line[key]["kwargs"]
    axes.plot(arc_from_start_to_back.Tx.value, arc_from_start_to_back.Ty.value,
              **kwargs)

# Add a line that indicates where the best Long score is
axes.plot(long_score_argmax_arc_from_start_to_back.Tx.value,
          long_score_argmax_arc_from_start_to_back.Ty.value, **bls_kwargs)

# Add a small circle to indicate the estimated epicenter of the wave
epicenter = Circle((initiation_point.Tx.value, initiation_point.Ty.value),
                   **epicenter_kwargs)
axes.add_patch(epicenter)

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
directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_wave_progress_map.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.savefig(full_file_path)


################################################################################
# Fit participation plot
# Plot and save a composite map with the following features.
# (1) Inverted b/w image of the Sun
# (2) Fit participation progress map
# (3) Colorbar with timestamps corresponding to the progress of the wave
# (4) Outlined circle showing the location of the putative wave source
# (5) Best long score arc indicated

# Create the composite map
c_map = Map(sun_image, fit_participation_map, composite=True)

# Create the figure
figure = plt.figure(2)
axes = figure.add_subplot(111)
ret = c_map.plot(axes=axes, title="fit participation map")
c_map.draw_limb(**draw_limb_kwargs)
c_map.draw_grid(**draw_grid_kwargs)

# Add a line that indicates where the best Long score is
axes.plot(long_score_argmax_arc_from_start_to_back.Tx.value,
          long_score_argmax_arc_from_start_to_back.Ty.value, **bls_kwargs)

# Add in lines that indicate 0, 90, 180 and 270 degrees
for key in line.keys():
    arc_from_start_to_back = extract[key][2]
    kwargs = line[key]["kwargs"]
    axes.plot(arc_from_start_to_back.Tx.value, arc_from_start_to_back.Ty.value,
              **kwargs)


# Add a small circle to indicate the estimated epicenter of the wave
epicenter = Circle((initiation_point.Tx.value, initiation_point.Ty.value),
                   **epicenter_kwargs)
axes.add_patch(epicenter)

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
directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_fit_participation_map.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.savefig(full_file_path)


################################################################################
# Plot and save the best long score arc
#
results[0][long_score_argmax][1].answer.plot(title='wave propagation at the best Long score\n(longitude={:s})'.format(bls_string))
plt.tight_layout()
directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_arc_with_highest_score.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.savefig(full_file_path)


###############################################################################
# Plot and save a map of the Long et al 2014 scores
figure = plt.figure(4)
axes = figure.add_subplot(111)

# Create the composite map
c_map = Map(sun_image, long_score_map, composite=True)
ret = c_map.plot(axes=axes, title="Long scores")
c_map.draw_limb(**draw_limb_kwargs)
c_map.draw_grid(**draw_grid_kwargs)

# Add a line that indicates where the best Long score is
axes.plot(long_score_argmax_arc_from_start_to_back.Tx.value,
          long_score_argmax_arc_from_start_to_back.Ty.value, **bls_kwargs)

# Add in lines that indicate 0, 90, 180 and 270 degrees
for key in line.keys():
    arc_from_start_to_back = extract[key][2]
    kwargs = line[key]["kwargs"]
    axes.plot(arc_from_start_to_back.Tx.value, arc_from_start_to_back.Ty.value,
              **kwargs)

# Add a small circle to indicate the estimated epicenter of the wave
epicenter = Circle((initiation_point.Tx.value, initiation_point.Ty.value),
                   **epicenter_kwargs)
axes.add_patch(epicenter)

# Add a colorbar
cbar = figure.colorbar(ret[1])
cbar.set_label('Long scores (%)')
cbar.set_clim(vmin=0.00, vmax=100.0)

# Save the map
directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_long_scores_map.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.savefig(full_file_path)


###############################################################################
# Fitted arcs progress map.  This is the closest in form to the plots shown
# in the Long et al paper.
#
figure = plt.figure(5)
axes = figure.add_subplot(111)

# Create the composite map
c_map = Map(sun_image, fitted_arcs_progress_map, composite=True)
ret = c_map.plot(axes=axes, title="wave progress along fitted arcs")
c_map.draw_limb(**draw_limb_kwargs)
c_map.draw_grid(**draw_grid_kwargs)


# Add a line that indicates where the best Long score is
axes.plot(long_score_argmax_arc_from_start_to_back.Tx.value,
          long_score_argmax_arc_from_start_to_back.Ty.value,  **bls_kwargs)

# Add in lines that indicate 0, 90, 180 and 270 degrees
for key in line.keys():
    arc_from_start_to_back = extract[key][2]
    kwargs = line[key]["kwargs"]
    axes.plot(arc_from_start_to_back.Tx.value, arc_from_start_to_back.Ty.value,
              **kwargs)

# Add a small circle to indicate the estimated epicenter of the wave
epicenter = Circle((initiation_point.Tx.value, initiation_point.Ty.value),
                   **epicenter_kwargs)
axes.add_patch(epicenter)

# Set up the color bar
nticks = 6
timestamps_index = np.linspace(1, len(timestamps)-1, nticks, dtype=np.int).tolist()
cbar_tick_labels = []
for index in timestamps_index:
    wpm_time = timestamps[index].strftime("%H:%M:%S")
    cbar_tick_labels.append(wpm_time)
cbar = figure.colorbar(ret[1])
cbar.set_label('seconds since {:s}'.format(observation_datetime))
cbar.set_clim(vmin=0, vmax=np.max(fitted_arcs_progress_map.data))


# Save the map
directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_fitted_arcs_progress_map.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.savefig(full_file_path)
