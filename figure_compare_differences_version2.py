#
# Script that loads in a data set and creates a series of plots
# that compare the effect of different running difference algorithms
# Version 2
#

import os
from copy import deepcopy

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LinearStretch

import mapcube_tools
import swave_study as sws
import aware_utils

filepath = '~/eitwave/img/difference_comparison.png'

# Summing of the simulated observations in the time direction
temporal_summing = sws.temporal_summing

# Summing of the simulated observations in the spatial directions
spatial_summing = sws.spatial_summing

# Which waves
wave_names = ['longetal2014_figure8a', 'longetal2014_figure8e', 'longetal2014_figure4']

# Differencing types
differencing_types = ['RDP', 'RD', 'BD', 'PBD']

# Plot limits that show off the differencing types nicely
emission_difference = 100
fractional_difference = 0.25


# indices
indices = {}
for wave_name in wave_names:
    indices[wave_name] = {}
    for differencing_type in differencing_types:
        if differencing_type in ('BD', 'PBD'):
            indices[wave_name][differencing_type] = 16
        else:
            indices[wave_name][differencing_type] = 15

# Storage for the maps
maps = {}

# Go through each wave
for i, wave_name in enumerate(wave_names):

    # Storage by wave name
    print("\n----------------")
    print("Loading and accumulating {:s} data".format(wave_name))
    maps[wave_name] = {}

    # Load observational data from file
    euv_wave_data = aware_utils.create_input_to_aware_for_test_observational_data(wave_name)

    # Accumulate the AIA data
    mc = euv_wave_data['finalmaps']
    mc = mapcube_tools.accumulate(mapcube_tools.superpixel(mc, spatial_summing), temporal_summing)

    # Go through each of the differencing types
    for differencing_type in differencing_types:

        # Which layer in the mapcube to use
        index = indices[wave_name][differencing_type]

        if differencing_type == 'RD':
            mc_diff = mapcube_tools.running_difference(mc)
        elif differencing_type == 'BD':
            mc_diff = mapcube_tools.base_difference(mc)
        elif differencing_type == 'RDP':
            mc_diff = mapcube_tools.running_difference(mapcube_tools.persistence(mc))
        elif differencing_type == 'PBD':
            mc_diff = mapcube_tools.base_difference(mc, fraction=True)
        else:
            raise ValueError('Unknown differencing type')

        # Store the maps
        maps[wave_name][differencing_type] = deepcopy(mc_diff[index])

        # What time is this?
        print('\n{:s} - {:s}'.format(wave_name, differencing_type))
        print("Selected map is at time " + str(maps[wave_name][differencing_type].date))
        print("Index is {:n}".format(index))

        # How many maps are in the map cube
        print('Number of maps = {:n}'.format(len(mc_diff)))


# Post processing - set colors etc

# Change the stretching and limits etc
cmap = cm.RdBu
for wave_name in wave_names:
    for differencing_type in differencing_types:
        this_map = maps[wave_name][differencing_type]
        this_map.plot_settings['norm'] = ImageNormalize(stretch=LinearStretch())
        if differencing_type == 'PBD':
            not_finite = ~np.isfinite(this_map.data)
            this_map.data[not_finite] = 0.0
            too_big = np.abs(this_map.data) > 1
            this_map.data[too_big] = 1
            this_map.plot_settings['norm'].vmax = fractional_difference
            this_map.plot_settings['norm'].vmin = -fractional_difference
            this_map.plot_settings['cmap'] = cmap
        else:
            this_map.plot_settings['norm'].vmax = emission_difference
            this_map.plot_settings['norm'].vmin = -emission_difference
            this_map.plot_settings['cmap'] = cmap

# Go through each wave
for wave_name in wave_names:

    # Go through each differencing type
    for differencing_type in differencing_types:

        # Get the map
        m = maps[wave_name][differencing_type]

        # New image
        fig, ax = plt.subplots()
        m.plot(axes=ax, title=differencing_type + '\n' + m.date.strftime("%Y/%m/%d %H:%M:%S"))
        ax.axes.yaxis.set_visible(False)
        ax.axes.xaxis.set_visible(False)
        plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.0, 0.0, 1.0, 1.0))
        f = os.path.expanduser(filepath + '.' + wave_name+ '.' + differencing_type + '.png')
        plt.savefig(f)
        plt.close('all')

titles = ['Wave A', 'Wave B', 'Wave C']
fig, axes = plt.subplots(4, 3, figsize=(8, 12))
# Go through each differencing type
for j, differencing_type in enumerate(differencing_types):

    for i, wave_name in enumerate(wave_names):

        # Get the map
        m = maps[wave_name][differencing_type]

        # New image
        ax = axes[j, i]

        if j == 0:
            title = titles[i]
        else:
            title = None

        m.plot(axes=ax, title=title)
        ax.axes.xaxis.set_visible(False)
        if j == 0:
            ax.set_title(title, fontsize=20)

        if i == 0:
            ax.set_ylabel(differencing_type, fontsize=20)
            ax.set_yticks([])
        else:
            ax.axes.yaxis.set_visible(False)


plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.0, 0.0, 1.0, 1.0))
f = os.path.expanduser(filepath)
plt.savefig(f)
plt.close('all')
