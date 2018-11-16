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
import matplotlib
from matplotlib.ticker import NullFormatter

from skimage.morphology import binary_dilation, disk


import astropy.units as u
from astropy.visualization import LinearStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.coordinates import SkyCoord

from sunpy.map import Map
from sunpy.time import parse_time
import sunpy.coordinates
from sunpy.coordinates import frames

import statistics_tools

# Main AWARE processing and detection code
import aware5_without_swapping_emission_axis

# Extra utilities for AWARE
import aware_utils

# Plotting stuff
import aware_plot
from aware_plot import longitudinal_lines
from aware_constants import solar_circumference_per_degree_in_km

# Mapcube handling tools
import mapcube_tools

# Code to create a simulated wave
from sim.wave2d import wave2d

# Parameters for the simulated wave
import swave_params

# Details of the study
import swave_study as sws

# Output fontsize
matplotlib.rcParams.update({'font.size': 18})

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


def coordinates_along_arc(great_circle, start_angle, end_angle):
    """
    Find the co-ordinates long an arc between the start angle
    and the end angle.

    great_circle
    start_angle
    end_angle
    """
    innner_angles = great_circle.inner_angles()
    diff_min = np.argmin(np.abs(innner_angles - start_angle))
    diff_max = np.argmin(np.abs(innner_angles - end_angle))
    return great_circle.coordinates()[diff_min:diff_max]


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
                                            output=['finalmaps', 'raw', 'transformed', 'noise'],
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

        # Get the final map out from the wave simulation and apply the accummulations
        mc = mapcube_tools.accumulate(mapcube_tools.superpixel(euv_wave_data[analysis_data_sources], spatial_summing), temporal_summing)
    else:
        # Load observational data from file
        print('Loading observational data - {:s}'.format(wave_name))
        euv_wave_data = aware_utils.create_input_to_aware_for_test_observational_data(wave_name, spatial_summing, temporal_summing)
        mc = euv_wave_data['finalmaps']
        # Transform parameters used to convert HPC image data to HG data.
        # The HPC data is transformed to HG using the location below as the
        # "pole" around which the data is transformed
        transform_hpc2hg_parameters['epi_lon'] = euv_wave_data['epi_lon'] * u.deg
        transform_hpc2hg_parameters['epi_lat'] = euv_wave_data['epi_lat'] * u.deg

    # Storage for the results from all methods and polynomial fits
    print(' - Using the griddata method %s.' % griddata_method)

    # If more than one randomization is requested for observational data
    # make a noisy realization of the observed data
    if observational and n_random > 1:
        print(' - Randomizing the observational data')
        mc = mapcube_tools.mapcube_noisy_realization(mc)

    ############################################################################
    # Initial map that shows an image of the Sun.  First data used in all
    # further analysis
    initial_map = deepcopy(mc[0])

    ############################################################################
    # Define the estimated initiation point
    initiation_point = SkyCoord(transform_hpc2hg_parameters['epi_lon'], transform_hpc2hg_parameters['epi_lat'],
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
                                                                           radii=radii,
                                                                           func=intensity_scaling_function,
                                                                           histogram_clip=histogram_clip,
                                                                           returned=['clipped'])
        print(' - Segmenting the data to get the emission due to wavefront')
        progress_mask_cube = aware_utils.progress_mask(aware_processed['cleaned'])
        segmented_maps = mapcube_tools.multiply(progress_mask_cube, aware_processed['clipped'])

        # Times of the data
        times = [m.date for m in segmented_maps]

        # Map for locations that participate in the fit
        fit_participation_datacube = np.zeros_like(segmented_maps.as_array())

        # Calculate the arcs
        print(' - Creating arc location information')
        # Number of arcs
        nlon = 360

        # Equally spaced arcs
        angles = (np.linspace(0, 2*np.pi, nlon+1))[0:-1] * u.rad

        # Number of files that we are examining
        nt = len(segmented_maps)

        # Define the great circles. The first great circle at position zero is the one that is most
        # closely directed towards the solar north pole.
        great_circles = aware_utils.great_circles_from_initiation_to_north_pole(initiation_point, initial_map, angles, great_circle_points)

        # Extract information from the great circles
        extract = aware_utils.extract_from_great_circles(great_circles, initial_map)

        # Detailed great circles for plotting
        great_circles_detailed = aware_utils.great_circles_from_initiation_to_north_pole(initiation_point, initial_map, angles, 100000)
        extract_detailed = aware_utils.extract_from_great_circles(great_circles_detailed, initial_map)

        print(' - Fitting polynomials to arcs')

        # Storage for fitting results at each longitude
        longitude_fit = []

        # Go through all the longitudes aka great circles
        for lon in range(0, nlon):
            # Get the Great Circle information
            this_great_circle = great_circles[lon]
            these_inner_angles = this_great_circle.inner_angles()
            these_pixel_coordinates_x, these_pixel_coordinates_y = this_great_circle.coordinates().to_pixel(initial_map.wcs)

            # At each longitude extract the data as required
            lat_time_data = aware5_without_swapping_emission_axis.build_lat_time_data(lon, extract, segmented_maps)

            # Define the next arc
            pixels = extract[lon][0]
            latitude = extract[lon][1]
            arc = aware5_without_swapping_emission_axis.Arc(lat_time_data, times, latitude, angles[lon].to(u.deg),
                                                            start_time=initial_map.date, sigma=np.sqrt(lat_time_data))

            # Measure the location of the wave and estimate an error in its location
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
                analysis.answer = aware5_without_swapping_emission_axis.FitPosition(arc.t, position, position_error,
                                                                                    ransac_kwargs=ransac_kwargs,
                                                                                    fit_method=fit_method,
                                                                                    n_degree=n_degree,
                                                                                    arc_identity=arc.longitude,
                                                                                    error_tolerance_kwargs=error_tolerance_kwargs)
                # Store each polynomial degree
                polynomial_degree_fit.append(analysis)

                # Update the fit participation mask
                if analysis.answer.fitted:
                    # Find which time indices were fitted
                    time_indices_fitted = analysis.answer.indicesf

                    # Find the location at each time index
                    degrees_from_initiation = analysis.answer.locf

                    # Go through each of the time indices
                    for k in range(0, len(time_indices_fitted)):
                        # Specific time index
                        tif = time_indices_fitted[k]

                        # Where the wave was at the time index
                        dfi = degrees_from_initiation[k] * u.deg

                        # Get the x, y position in a map corresponding to where
                        # the wave was along the arc.
                        angular_index = np.argmin(np.abs(these_inner_angles - dfi))
                        x = int(np.rint(these_pixel_coordinates_x[angular_index]))
                        y = int(np.rint(these_pixel_coordinates_y[angular_index]))

                        # Fill in the fit participation datacube
                        fit_participation_datacube[y, x, tif] = 1

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
epicenter_kwargs = {"edgecolor": 'w', "facecolor": "c", "radius": 50, "fill": True, "zorder": 1000}

# Guide lines on the sphere
line = aware_plot.longitudinal_lines

# Long score formatting
bls_kwargs = {"color": "r", "zorder": 1002, "linewidth": 3}
bls_error_kwargs = {"color": "red", "zorder": 1001, "linewidth": 1}
fitted_arc_kwargs = {"linewidth": 1, "color": 'b'}

# True values for non-observational data
zorder_min = 10000000
true_velocity_kwargs = {"color": "blue", "linewidth": 3, "linestyle": "-.", "zorder": zorder_min+1}
true_acceleration_kwargs = {"color": "red", "linewidth": 3, "linestyle": "-.", "zorder": zorder_min+1}

# Measured values for the scatter plots
# Long et el range values
lr_kwargs = {"color": "black", "linestyle": ":"}

velocity_median_kwargs = {"color": "cyan", "linestyle": "--", "linewidth": 3, "zorder": zorder_min+1}
velocity_percentile_kwargs = {"color": "cyan", "linestyle": ":", "linewidth": 3, "zorder": zorder_min+1}

acceleration_median_kwargs = {"color": "orange", "linestyle": "--", "linewidth": 3, "zorder": zorder_min+1}
acceleration_percentile_kwargs = {"color": "orange", "linestyle": ":", "linewidth": 3, "zorder": zorder_min+1}

corpita_median_kwargs = {"color": "magenta", "linestyle": "--", "linewidth": 3, "zorder": zorder_min+1}
corpita_percentile_kwargs = {"color": "magenta", "linestyle": ":", "linewidth": 3, "zorder": zorder_min+1}

exceed_kwargs = {"linewidth": 1, "alpha": 1.0, "color": 'orange'}


# Legend keywords
legend_kwargs = {"framealpha": 0.7, "facecolor": "yellow", "loc": "best", "fontsize": 15}

# Image of the Sun used as a background
sun_image = deepcopy(initial_map)
sun_image.plot_settings['cmap'] = base_cm_sun_image
observation_date = initial_map.date.strftime("%Y-%m-%d")
observation_datetime = initial_map.date.strftime("%Y-%m-%d %H:%M:%S")
#
image_file_type = sws.image_file_type
observation = r"AIA {:s}".format(initial_map.measurement._repr_latex_())
#
longscorename = "CorPITA score"

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
    # Initial speed, acceleration and date
    speed = simulated_wave_parameters['speed'][0]
    acceleration = simulated_wave_parameters['acceleration']
    d0 = parse_time(euv_wave_data['finalmaps'][0].date)

    # Seconds since the initial 
    time = np.asarray([(parse_time(m.date) - d0).total_seconds() for m in euv_wave_data['finalmaps']]) * u.s
    true_position = speed * time + 0.5 * acceleration * time * time
    simulated_line = {"t": time, "y": true_position, "kwargs": {"label": "true position"}}

    # Velocity in km/s and their rendering in plots
    v_true = speed * solar_circumference_per_degree_in_km
    v_unit = v_true.unit.to_string('latex_inline')
    v_true_string = r'$v_{true}$'
    v_true_full = '{:s}={:.2f}{:s}'.format(v_true_string, v_true.value, v_unit)

    # Acceleration in km/s/s and their rendering in plots
    a_true = acceleration * solar_circumference_per_degree_in_km
    a_unit = a_true.unit.to_string('latex_inline')
    a_true_string = r'$a_{true}$'
    a_true_full = '{:s}={:.2f}{:s}'.format(a_true_string, a_true.value, a_unit)

else:
    v_unit = (1 * u.km/u.s).unit.to_string('latex_inline')
    a_unit = (1 * u.km/u.s/u.s).unit.to_string('latex_inline')

################################################################################
#
figure_label_longscore = longscorename
figure_label_bls = "wave propagation at best {:s}".format(longscorename)
figure_labels = {"wave progress map": "(a) ",
                 "fit participation map": "(b) ",
                 figure_label_longscore: "(c) ",
                 "wave progress along fitted arcs": "(d) ",
                 "fitted velocity": "(e) ",
                 "fitted acceleration": "(f) ",
                 figure_label_bls: "(g) "}

################################################################################
# Create the wave progress map
#
wave_progress_map, timestamps = aware_utils.wave_progress_map_by_location(aware_processed['cleaned'])
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

fit_participation_map_mask_data = np.sum(fit_participation_datacube, axis=2)
fit_participation_map_mask_data[fit_participation_map_mask_data > 0] = 1
fit_participation_map_mask_data = 1.0*binary_dilation(fit_participation_map_mask_data, selem=disk(1))
fit_participation_map_mask_data *= 0.25*len(timestamps)
fit_participation_map_mask = Map(1.0*fit_participation_map_mask_data, fit_participation_map.meta)
fit_participation_map_mask.plot_settings['cmap'] = wave_progress_map_cm
fit_participation_map_mask.plot_settings['norm'] = wave_progress_map_norm

################################################################################
# Create the long score data

# Long score
long_score = np.asarray([aaa[1].answer.long_score.final_score if aaa[1].answer.fitted else 0.0 for aaa in results[0]])

# Best Long score
long_score_argmax = long_score.argmax()

bls_string = (angles[long_score_argmax].to(u.deg))._repr_latex_()

# Make the map data
long_score_map = deepcopy(initial_map)
long_score_map.data[:, :] = -1.0
for lon in range(0, nlon):
    pixels_detailed = extract_detailed[lon][0]
    x = pixels_detailed[0, :]
    y = pixels_detailed[1, :]
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


################################################################################
# Find the extent of all the arcs fit and make a map of that

# Mask that will hold the fitted arcs
fitted_arcs_mask = np.zeros_like(long_score_map.data) - 1

# Go through all the longitudes
for lon in range(0, nlon):
    # Next fit
    answer = results[0][lon][1].answer

    # Maximum latitudinal extent
    if answer.fitted:
        answer_max_latitudinal_extent = answer.best_fit[-1] * u.deg
        answer_min_latitudinal_extent = answer.best_fit[0] * u.deg

        # Get the latitude of the arc
        latitude_detailed = extract_detailed[lon][1]

        # Get the pixels along the arc
        pixels_detailed = extract_detailed[lon][0]

        # Find the index where the arc latitude equals the maximum latitudinal
        # extent of the fit
        max_arg_latitude_detailed = np.argmin(np.abs(latitude_detailed - answer_max_latitudinal_extent))
        min_arg_latitude_detailed = np.argmin(np.abs(latitude_detailed - answer_min_latitudinal_extent))

        # Get the x and y pixels of the fitted arc and fill in the arc.
        x = pixels_detailed[0, min_arg_latitude_detailed:max_arg_latitude_detailed]
        y = pixels_detailed[1, min_arg_latitude_detailed:max_arg_latitude_detailed]

        # Calculate the time at all the latitudes
        bfp = answer.estimate
        if answer.n_degree == 2:
            z2 = bfp[1]**2 - 4*bfp[0]*(bfp[2] - latitude_detailed[min_arg_latitude_detailed:max_arg_latitude_detailed].value)
            fitted_arc_time = (-bfp[1] + np.sqrt(z2))/(2*bfp[0])
        else:
            fitted_arc_time = (latitude[min_arg_latitude_detailed:max_arg_latitude_detailed].value - bfp[1])/bfp[0]
        # Return in units of the summation
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
# Update the map of the Long Score
#
# Restrict to where the best fit says the wave was detected
fitted_arcs_mask2 = np.zeros_like(fitted_arcs_mask)
zero_and_above = fitted_arcs_mask >= 0
below_zero = fitted_arcs_mask < 0
fitted_arcs_mask2[zero_and_above] = 1
long_score_map.data *= fitted_arcs_mask2
long_score_map.data[below_zero] = -1

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
# bls_coordinates = extract[long_score_argmax][2][0:long_score_argmax_pixels_nonzero_index]

bls_answer = results[0][long_score_argmax][1].answer
bls_coordinates = coordinates_along_arc(great_circles[long_score_argmax],
                                        bls_answer.best_fit[0] * u.deg,
                                        bls_answer.best_fit[-1] * u.deg)
bls_error_coordinates = coordinates_along_arc(great_circles[long_score_argmax],
                                              np.min(bls_answer.best_fit_error[0]) * u.deg,
                                              np.max(bls_answer.best_fit_error[-1]) * u.deg)


###############################################################################
def lines_0_90_180_270(line, extract, axes):
    """
    Add lines to existing axes

    line: lines
    extract: pixel locations fof the lines
    axes: the Axes instance that the lines are drawn on

    """
    for key in line.keys():
        arc_from_start_to_back = extract[key][2]
        axes.plot(arc_from_start_to_back.Tx.value, arc_from_start_to_back.Ty.value, **line[key]["kwargs"])


###############################################################################
# Wave progress plot
# Plot and save a composite map with the following features.
# (1) Inverted b/w image of the Sun
# (2) Full on/off disk wave progress map
# (3) Colorbar with timestamps corresponding to the progress of the wave
# (4) Outlined circle showing the location of the putative wave source

this_figure = "wave progress map"

# Create the composite map
c_map = Map(sun_image, wave_progress_map, composite=True)

# Create the figure
figure = plt.figure(1)
axes = figure.add_subplot(111)
ret = c_map.plot(axes=axes, title="{:s}{:s}".format(figure_labels[this_figure], this_figure))
c_map.draw_limb(**draw_limb_kwargs)
c_map.draw_grid(**draw_grid_kwargs)
axes.grid('on', linestyle=":")

# Add in lines that indicate 0, 90, 180 and 270 degrees
lines_0_90_180_270(line, extract, axes)

# Add a line that indicates where the best Long score is
axes.plot(bls_coordinates.Tx.value, bls_coordinates.Ty.value, **bls_kwargs)
axes.plot(bls_error_coordinates.Tx.value, bls_error_coordinates.Ty.value, **bls_error_kwargs)

# Add a small circle to indicate the estimated epicenter of the wave
epicenter = Circle((initiation_point.Tx.value, initiation_point.Ty.value), **epicenter_kwargs)
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
plt.tight_layout()
plt.savefig(full_file_path)


################################################################################
# Fit participation plot
# Plot and save a composite map with the following features.
# (1) Inverted b/w image of the Sun
# (2) Fit participation progress map
# (3) Colorbar with timestamps corresponding to the progress of the wave
# (4) Outlined circle showing the location of the putative wave source
# (5) Best long score arc indicated

this_figure = "fit participation map"

# Create the composite map
c_map = Map(sun_image, fit_participation_map_mask, composite=True)

# Create the figure
figure = plt.figure(2)
axes = figure.add_subplot(111)
ret = c_map.plot(axes=axes, title="{:s}{:s}".format(figure_labels[this_figure], this_figure))
c_map.draw_limb(**draw_limb_kwargs)
c_map.draw_grid(**draw_grid_kwargs)
axes.grid('on', linestyle=":")

# Add a line that indicates where the best Long score is
axes.plot(bls_coordinates.Tx.value, bls_coordinates.Ty.value, **bls_kwargs)
axes.plot(bls_error_coordinates.Tx.value, bls_error_coordinates.Ty.value, **bls_error_kwargs)

# Add in lines that indicate 0, 90, 180 and 270 degrees
lines_0_90_180_270(line, extract, axes)

# Add a small circle to indicate the estimated epicenter of the wave
epicenter = Circle((initiation_point.Tx.value, initiation_point.Ty.value), **epicenter_kwargs)
axes.add_patch(epicenter)

# Set up the color bar
#nticks = 6
#timestamps_index = np.linspace(1, len(timestamps)-1, nticks, dtype=np.int).tolist()
#cbar_tick_labels = []
#for index in timestamps_index:
#    wpm_time = timestamps[index].strftime("%H:%M:%S")
#    cbar_tick_labels.append(wpm_time)
#cbar = figure.colorbar(ret[1], ticks=timestamps_index)
#cbar.ax.set_yticklabels(cbar_tick_labels)
#cbar.set_label('time (UT) ({:s})'.format(observation_date))
#cbar.set_clim(vmin=1, vmax=len(timestamps))

# Save the wave progress map
directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_fit_participation_map.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.tight_layout()
plt.savefig(full_file_path)


################################################################################
# Plot and save the best long score arc
#
this_figure = figure_label_bls
results[0][long_score_argmax][1].answer.plot(title='{:s}{:s}\n(longitude={:s})'.format(figure_labels[this_figure], this_figure, bls_string))
plt.tight_layout()
directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_arc_with_highest_score.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.tight_layout()
plt.savefig(full_file_path)


###############################################################################
# Plot and save a map of the Long et al 2014 scores
this_figure = figure_label_longscore
figure = plt.figure(4)
axes = figure.add_subplot(111)

# Create the composite map
c_map = Map(sun_image, long_score_map, composite=True)
ret = c_map.plot(axes=axes, title="{:s}{:s}".format(figure_labels[this_figure], this_figure))
c_map.draw_limb(**draw_limb_kwargs)
c_map.draw_grid(**draw_grid_kwargs)
axes.grid('on', linestyle=":")

# Add a line that indicates where the best Long score is
axes.plot(bls_coordinates.Tx.value, bls_coordinates.Ty.value, **bls_kwargs)
axes.plot(bls_error_coordinates.Tx.value, bls_error_coordinates.Ty.value, **bls_error_kwargs)

# Add in lines that indicate 0, 90, 180 and 270 degrees
lines_0_90_180_270(line, extract, axes)

# Add a small circle to indicate the estimated epicenter of the wave
epicenter = Circle((initiation_point.Tx.value, initiation_point.Ty.value), **epicenter_kwargs)
axes.add_patch(epicenter)

# Add a colorbar
cbar = figure.colorbar(ret[1])
cbar.set_label('{:s} (%)'.format(longscorename))
cbar.set_clim(vmin=0.00, vmax=100.0)

# Save the map
directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_long_scores_map.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.tight_layout()
plt.savefig(full_file_path)


###############################################################################
# Fitted arcs progress map.  This is the closest in form to the plots shown
# in the Long et al paper.
#
this_figure = "wave progress along fitted arcs"
figure = plt.figure(5)
axes = figure.add_subplot(111)

# Create the composite map
c_map = Map(sun_image, fitted_arcs_progress_map, composite=True)
ret = c_map.plot(axes=axes, title="{:s}{:s}".format(figure_labels[this_figure], this_figure))
c_map.draw_limb(**draw_limb_kwargs)
c_map.draw_grid(**draw_grid_kwargs)
axes.grid('on', linestyle=":")


# Add a line that indicates where the best Long score is
axes.plot(bls_coordinates.Tx.value, bls_coordinates.Ty.value, **bls_kwargs)
axes.plot(bls_error_coordinates.Tx.value, bls_error_coordinates.Ty.value, **bls_error_kwargs)

# Add in lines that indicate 0, 90, 180 and 270 degrees
lines_0_90_180_270(line, extract, axes)

# Add a small circle to indicate the estimated epicenter of the wave
epicenter = Circle((initiation_point.Tx.value, initiation_point.Ty.value), **epicenter_kwargs)
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
plt.tight_layout()
plt.savefig(full_file_path)


###############################################################################
# Get the velocities and accelerations and their errors, and also where the
# data was fit
#
deg_fit = []
deg_no_fit = []
deg_fit_bool = []
v = []
ve = []
a = []
ae = []
for lon, result in enumerate(results[0]):
    ra = result[1].answer
    deg_fit_bool.append(ra.fitted)
    if ra.fitted:
        deg_fit.append(lon)
        v.append((ra.velocity*solar_circumference_per_degree_in_km).value)
        ve.append((ra.velocity_error*solar_circumference_per_degree_in_km).value)
        a.append((ra.acceleration*solar_circumference_per_degree_in_km).value)
        ae.append((ra.acceleration_error*solar_circumference_per_degree_in_km).value)
    else:
        deg_no_fit.append(lon)
deg_fit = np.asarray(deg_fit)
deg_no_fit = np.asarray(deg_no_fit)
v = np.asarray(v)
ve = np.asarray(ve)
a = np.asarray(a)
ae = np.asarray(ae)
deg_fit_bool = np.asarray(deg_fit_bool)


###############################################################################
# Arc velocity plot.  Plots the velocity along all the arcs, along with their
# standard error
#
this_figure = 'fitted velocity'
v_long_range = [0.0, 2000.0]
a_fit = '$a_{fit}$'
v_fit = '$v_{fit}$'

longitudinal_lines_kwargs = {"bbox": dict(facecolor='yellow', alpha=1.0),
                             "fontsize": 9,
                             "horizontalalignment": 'center',
                             }
fig = plt.figure(6)
ax = fig.add_subplot(111)

# Plot the found initial velocities
ax.errorbar(deg_fit, v, yerr=ve, color='green', label='{:s}'.format(v_fit), linewidth=0.5, fmt='o', alpha=1.0, markersize=5)
ax.xaxis.set_ticks(np.arange(0, 360, 45))
# Plot the true velocity if not observational
if not observational:
    ax.axhline(v_true.value, label=v_true_full, **true_velocity_kwargs)

# Axis labels and titles
ax.set_xlabel('longitude (degrees)')
ax.set_ylabel('velocity ($km s^{-1}$)')
ax.set_title('{:s}{:s} {:s}'.format(figure_labels[this_figure], this_figure, v_fit))
ax.grid('on', linestyle=':')
ax.set_ylim(v_long_range[0], np.min([v_long_range[1], np.max(v)]))
for key in longitudinal_lines.keys():
    ax.axvline(key, **longitudinal_lines[key]['kwargs'])
dynamics_bls_label = 'best {:s} ({:n}{:s})'.format(longscorename, long_score_argmax, u.degree.to_string('latex_inline'))
ax.axvline(long_score_argmax, color='red', label=dynamics_bls_label)
#for i in range(0, len(deg_no_fit)):
#    if i == 0:
#        ax.axvline(deg_no_fit[i], linewidth=0.5, alpha=0.5, color='blue', label='no fit')
#    else:
#        ax.axvline(deg_no_fit[i], linewidth=0.5, alpha=0.5, color='blue')
first_flag = True
for i in range(0, len(deg_fit)):
    if v[i] < v_long_range[0] or v[i] > v_long_range[1]:
        if first_flag:
            ax.axvline(deg_fit[i], label='{:s}<{:.0f}, {:s}>{:.0f}'.format(v_fit, v_long_range[0], v_fit, v_long_range[1]), **exceed_kwargs)
            first_flag = False
        else:
            ax.axvline(deg_fit[i], **exceed_kwargs)

l = plt.legend(**legend_kwargs)
l.set_zorder(10*zorder_min)
# Save the plot
directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_velocity_longitude_plot.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.tight_layout()
plt.savefig(full_file_path)

# Print out the number, mean, standard deviation and range of the velocity
print('Stats for all found initial velocities')
print('Number of arcs found = {:n}'.format(len(v)))
print('Mean of the initial velocity = {:n}'.format(np.mean(v)))
print('Standard deviation of the initial velocity = {:n}'.format(np.std(v)))
v_above_lo = v > v_long_range[0]
v_below_hi = v < v_long_range[1]
v_limited = np.logical_and(v_above_lo, v_below_hi)
print('Stats for all found initial velocities within the Long limits')
print('Number of arcs found = {:n}'.format(len(v[v_limited])))
print('Mean of the initial velocity = {:n}'.format(np.mean(v[v_limited])))
print('Standard deviation of the initial velocity = {:n}'.format(np.std(v[v_limited])))


###############################################################################
# Arc acceleration plot.  Plots the velocity along all the arcs, along with
# their standard error
#
this_figure = 'fitted acceleration'
a_long_range = [-2.0, 2.0]
fig = plt.figure(7)
ax = fig.add_subplot(111)

# Plot the found acceleration
ax.errorbar(deg_fit, a, yerr=ae, color='green', label='{:s}'.format(a_fit), linewidth=0.5, fmt='o', alpha=1.0, markersize=5)
ax.xaxis.set_ticks(np.arange(0, 360, 45))
# Plot the true acceleration if not observational
if not observational:
    ax.axhline(a_true.value, label=a_true_full, **true_acceleration_kwargs)

# Axis labels and titles
ax.set_xlabel('longitude (degrees)')
ax.set_ylabel('acceleration ($km s^{-2}$)')
ax.set_title('{:s}{:s} {:s}'.format(figure_labels[this_figure], this_figure, a_fit))
ax.grid('on', linestyle=':')
ax.set_ylim(np.max([a_long_range[0], np.min(a)]), np.min([a_long_range[1], np.max(a)]))
for key in longitudinal_lines.keys():
    ax.axvline(key, **longitudinal_lines[key]['kwargs'])
ax.axvline(long_score_argmax, color='red', label=dynamics_bls_label)
#for i in range(0, len(deg_no_fit)):
#    if i == 0:
#        ax.axvline(deg_no_fit[i], linewidth=0.5, alpha=0.5, color='blue', label='no fit')
#    else:
#        ax.axvline(deg_no_fit[i], linewidth=0.5, alpha=0.5, color='blue')
first_flag = True
for i in range(0, len(deg_fit)):
    if a[i] < a_long_range[0] or a[i] > a_long_range[1]:
        if first_flag:
            ax.axvline(deg_fit[i], label='{:s}<{:.0f}, {:s}>{:.0f}'.format(a_fit, a_long_range[0], a_fit, a_long_range[1]), **exceed_kwargs)
            first_flag = False
        else:
            ax.axvline(deg_fit[i], **exceed_kwargs)

l = plt.legend(**legend_kwargs)
l.set_zorder(10*zorder_min)
# Save the plot
directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_acceleration_longitude_plot.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.tight_layout()
plt.savefig(full_file_path)


def scatter_results_label(s, name, unit="", fmt="{:.1f}"):
    q_range = "{:.1f}, {:.1f}% {:s}".format(s.q[0], s.q[1], name)
    results = "({:.1f}, {:.1f}{:s})".format(s.percentile[0], s.percentile[1], unit)
    return "{:s} {:s}".format(q_range, results)


###############################################################################
# Plots of velocity versus acceleration
# scatter plot and histograms
#
ls = long_score[deg_fit_bool]
central_range = 68.0
percentile_range = [0.5*(100-central_range), 100-0.5*(100-central_range)]
v_summary = statistics_tools.Summary(v, q=percentile_range)
a_summary = statistics_tools.Summary(a, q=percentile_range)
ls_summary = statistics_tools.Summary(ls, q=percentile_range)
n_fit = len(ls)
n_fit_string = "{:n} out of 360".format(n_fit)

v_median_string = "median {:s} ({:.1f}{:s})".format(v_fit, v_summary.median, v_unit)
a_median_string = "median {:s} ({:.1f}{:s})".format(a_fit, a_summary.median, a_unit)
ls_median_string = "median {:s} ({:.1f})".format(longscorename, ls_summary.median)


figsize = (16, 8)
nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.6
bottom, height = 0.1, 0.6
bottom_h = left_h = left + width + 0.03

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(8, figsize=figsize)

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.errorbar(v, a, xerr=ve, yerr=ae, elinewidth=0.5, ecolor='k', marker='o', markeredgecolor='k', fmt='o', capsize=1)
#axScatter.scatter(v, a, edgecolor='k', s=np.abs())
axScatter.set_xlabel(r'fitted velocity ($km s^{{{-1}}}$)')
axScatter.set_ylabel(r'fitted acceleration ($km s^{-2}$)')
axScatter.grid('on', linestyle=":")
#axScatter.axvline(v_long_range[0], **lr_kwargs)
#axScatter.axvline(v_long_range[1], **lr_kwargs)
axScatter.axvline(v_summary.median, label=v_median_string, **velocity_median_kwargs)
axScatter.axvline(v_summary.percentile[0], label=scatter_results_label(v_summary, v_fit, unit=v_unit), **velocity_percentile_kwargs)
axScatter.axvline(v_summary.percentile[1], **velocity_percentile_kwargs)

#axScatter.axhline(a_long_range[0], **lr_kwargs)
#axScatter.axhline(a_long_range[1], **lr_kwargs)
axScatter.axhline(a_summary.median,  label=a_median_string, **acceleration_median_kwargs)
axScatter.axhline(a_summary.percentile[0], label=scatter_results_label(a_summary, a_fit, unit=a_unit), **acceleration_percentile_kwargs)
axScatter.axhline(a_summary.percentile[1], **acceleration_percentile_kwargs)

axScatter.set_xlim((np.min(v), np.max(v)))
axScatter.set_ylim((np.min(a), np.max(a)))

if not observational:
    axScatter.axvline(v_true.value, label=v_true_full, **true_velocity_kwargs)
    axScatter.axhline(a_true.value, label=a_true_full, **true_acceleration_kwargs)

# Add a legend
l = axScatter.legend(**legend_kwargs)
l.set_zorder(10*zorder_min)

xbins = 30
axHistx.hist(v, bins=xbins)
axHistx.set_ylabel('number')
axHistx.grid('on', linestyle=":")
axHistx.set_xlim(axScatter.get_xlim())
#axHistx.axvline(v_long_range[0], **lr_kwargs)
#axHistx.axvline(v_long_range[1], **lr_kwargs)
axHistx.axvline(v_summary.median, **velocity_median_kwargs)
axHistx.axvline(v_summary.percentile[0], **velocity_percentile_kwargs)
axHistx.axvline(v_summary.percentile[1], **velocity_percentile_kwargs)
axHistx.set_title('(a) {:s} vs. {:s} ({:s})'.format(v_fit, a_fit, n_fit_string))
if not observational:
    axHistx.axvline(v_true.value, **true_velocity_kwargs)


ybins = 30
axHisty.hist(a, bins=ybins, orientation='horizontal')
axHisty.set_xlabel('number')
axHisty.grid('on', linestyle=":")
axHisty.set_ylim(axScatter.get_ylim())
#axHisty.axhline(a_long_range[0], **lr_kwargs)
#axHisty.axhline(a_long_range[1], **lr_kwargs)
axHisty.axhline(a_summary.median, **acceleration_median_kwargs)
axHisty.axhline(a_summary.percentile[0], **acceleration_percentile_kwargs)
axHisty.axhline(a_summary.percentile[1], **acceleration_percentile_kwargs)
if not observational:
    axHisty.axhline(a_true.value, **true_acceleration_kwargs)


directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_a_v_scatter.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.tight_layout()
plt.savefig(full_file_path)


###############################################################################
# Plots of velocity and Long-score
# scatter plot and histograms
#

# start with a rectangular Figure
plt.figure(9, figsize=figsize)

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.errorbar(v, ls, xerr=ve, elinewidth=0.5, ecolor='k', marker='o', markeredgecolor='k', fmt='o', capsize=1)
axScatter.set_xlabel('fitted velocity ($km s^{-1}$)')
axScatter.set_ylabel(longscorename)
axScatter.grid('on', linestyle=":")
#axScatter.axvline(v_long_range[0], **lr_kwargs)
#axScatter.axvline(v_long_range[1], **lr_kwargs)
axScatter.axvline(v_summary.median, label=v_median_string, **velocity_median_kwargs)
axScatter.axvline(v_summary.percentile[0], label=scatter_results_label(v_summary, v_fit, unit=v_unit), **velocity_percentile_kwargs)
axScatter.axvline(v_summary.percentile[1], **velocity_percentile_kwargs)
axScatter.axhline(ls_summary.median, label=ls_median_string, **corpita_median_kwargs)
axScatter.axhline(ls_summary.percentile[0], label=scatter_results_label(ls_summary, longscorename), **corpita_percentile_kwargs)
axScatter.axhline(ls_summary.percentile[1], **corpita_percentile_kwargs)

axScatter.set_xlim((np.min(v), np.max(v)))
axScatter.set_ylim((0, 100))
if not observational:
    axScatter.axvline(v_true.value, label=v_true_full, **true_velocity_kwargs)

# Add a legend
l = axScatter.legend(**legend_kwargs)
l.set_zorder(10*zorder_min)

xbins = 30
axHistx.hist(v, bins=xbins)
axHistx.set_ylabel('number')
axHistx.grid('on', linestyle=":")
axHistx.set_xlim(axScatter.get_xlim())
#axHistx.axvline(v_long_range[0], **lr_kwargs)
#axHistx.axvline(v_long_range[1], **lr_kwargs)
axHistx.axvline(v_summary.median, **velocity_median_kwargs)
axHistx.axvline(v_summary.percentile[0], **velocity_percentile_kwargs)
axHistx.axvline(v_summary.percentile[1], **velocity_percentile_kwargs)
axHistx.set_title('(b) {:s} vs. {:s}'.format(v_fit, longscorename))
if not observational:
    axHistx.axvline(v_true.value, **true_velocity_kwargs)


ybins = 30
axHisty.hist(ls, bins=ybins, orientation='horizontal')
axHisty.set_xlabel('number')
axHisty.grid('on', linestyle=":")
axHisty.set_ylim(axScatter.get_ylim())
axHisty.axhline(ls_summary.median, **corpita_median_kwargs)
axHisty.axhline(ls_summary.percentile[0], **corpita_percentile_kwargs)
axHisty.axhline(ls_summary.percentile[1], **corpita_percentile_kwargs)

directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_v_ls_scatter.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.tight_layout()
plt.savefig(full_file_path)

###############################################################################
# Plots of acceleration and Long-score
# scatter plot and histograms
#

# start with a rectangular Figure
plt.figure(10, figsize=figsize)

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.errorbar(ls, a, yerr=ae, elinewidth=0.5, ecolor='k', marker='o', markeredgecolor='k', fmt='o', capsize=1)
axScatter.set_ylabel('fitted acceleration ($km s^{-2}$)')
axScatter.set_xlabel(longscorename)
axScatter.grid('on', linestyle=":")
axScatter.axvline(ls_summary.median, label=ls_median_string, **corpita_median_kwargs)
axScatter.axvline(ls_summary.percentile[0], label=scatter_results_label(ls_summary, longscorename), **corpita_percentile_kwargs)
axScatter.axvline(ls_summary.percentile[1], **corpita_percentile_kwargs)
axScatter.axhline(a_summary.median, label=a_median_string, **acceleration_median_kwargs)
axScatter.axhline(a_summary.percentile[0], label=scatter_results_label(a_summary, a_fit, unit=a_unit), **acceleration_percentile_kwargs)
axScatter.axhline(a_summary.percentile[1], **acceleration_percentile_kwargs)

axScatter.set_ylim((np.min(a), np.max(a)))
axScatter.set_xlim((0, 100))
if not observational:
    axScatter.axhline(a_true.value, label=a_true_full, **true_acceleration_kwargs)

# Add a legend
l = axScatter.legend(**legend_kwargs)
l.set_zorder(10*zorder_min)

xbins = 30
axHistx.hist(ls, bins=xbins)
axHistx.set_ylabel('number')
axHistx.grid('on', linestyle=":")
axHistx.set_xlim(axScatter.get_xlim())
axHistx.set_title('(c) {:s} vs. {:s}'.format(longscorename, a_fit))
axHistx.axvline(ls_summary.median, **corpita_median_kwargs)
axHistx.axvline(ls_summary.percentile[0], **corpita_percentile_kwargs)
axHistx.axvline(ls_summary.percentile[1], **corpita_percentile_kwargs)

ybins = 30
axHisty.hist(a, bins=ybins, orientation='horizontal')
axHisty.set_xlabel('number')
axHisty.grid('on', linestyle=":")
axHisty.set_ylim(axScatter.get_ylim())
axHisty.axhline(a_summary.median, **acceleration_median_kwargs)
axHisty.axhline(a_summary.percentile[0], **acceleration_percentile_kwargs)
axHisty.axhline(a_summary.percentile[1], **acceleration_percentile_kwargs)
if not observational:
    axHisty.axhline(a_true.value, **true_acceleration_kwargs)

directory = otypes_dir['img']
filename = aware_utils.clean_for_overleaf(otypes_filename['img']) + '_ls_a_scatter.{:s}'.format(image_file_type)
full_file_path = os.path.join(directory, filename)
plt.tight_layout()
plt.savefig(full_file_path)

