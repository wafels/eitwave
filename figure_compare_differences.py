#
# Script that loads in a data set and creates a series of plots
# that compare the effect of different running difference algorithms

import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mapcube_tools
import swave_study as sws
import aware_utils

filepath = '~/eitwave/img/difference_comparison.png'

# Summing of the simulated observations in the time direction
temporal_summing = sws.temporal_summing

# Summing of the simulated observations in the spatial directions
spatial_summing = sws.spatial_summing

wave_names = ['longetal2014_figure8a', 'longetal2014_figure8e', 'longetal2014_figure4']

differencing_types = ['RDP', 'RD', 'PBD']

info = {'longetal2014_figure8a': 20,
        'longetal2014_figure8e': 20,
        'longetal2014_figure4': 20}

fontsize = 8
maps = {}

figure_type = 2

#
# Set up the matplotlib plot
#
plt.close('all')
fig, axes = plt.subplots(3, len(wave_names), figsize=(9, 9))

# Go through each wave
for i, wave_name in enumerate(wave_names):

    index = info[wave_name]

    # Load observational data from file
    euv_wave_data = aware_utils.create_input_to_aware_for_test_observational_data(wave_name)

    #
    print('Accumulating AIA data.')
    mc = euv_wave_data['finalmaps']
    mc = mapcube_tools.accumulate(mapcube_tools.superpixel(mc, spatial_summing), temporal_summing)

    for differencing_type in differencing_types:
        if differencing_type == 'RD':
            # running difference
            print('Calculating the running difference.')
            mc_rd = mapcube_tools.running_difference(mc)
            new = deepcopy(mc_rd[index])
            new.plot_settings['cmap'] = cm.RdGy

        if differencing_type == 'PBD':
            # fraction base difference
            print('Calculating the base difference.')
            mc_pbd = mapcube_tools.base_difference(mc, fraction=True)
            new = deepcopy(mc_pbd[index+1])
            new.plot_settings['norm'].vmax = 0.5
            new.plot_settings['norm'].vmin = -0.5
            new.plot_settings['cmap'] = cm.RdGy

        if differencing_type == 'RDP':
            # running difference persistence images
            print('Calculating the running difference persistence images.')
            mc_rdp = mapcube_tools.running_difference(mapcube_tools.persistence(mc))
            new = deepcopy(mc_rdp[index])
            new.plot_settings['cmap'] = cm.gray_r

        maps[differencing_type] = new

    rd_all_vmax = np.max([maps['RD'].plot_settings['norm'].vmax,
                          maps['RDP'].plot_settings['norm'].vmax])
    maps['RD'].plot_settings['norm'].vmax = rd_all_vmax
    maps['RDP'].plot_settings['norm'].vmax = rd_all_vmax

    # Go through each differencing type
    for j, differencing_type in enumerate(differencing_types):
        tm = maps[differencing_type]
        ta = axes[j, i]

        # Just use the built in map plotting
        if figure_type == 1:
            if j == 0:
                tm.plot(axes=ta, title=differencing_type + '\n' + tm.date.strftime("%Y/%m/%d %H:%M:%S"))
            else:
                tm.plot(axes=ta, title=differencing_type)
            tm.draw_limb(color='black')
            ta.set_xlabel('x (arcsec)', fontsize=fontsize)
            xtl = ta.axes.xaxis.get_majorticklabels()
            for l in range(0, len(xtl)):
                xtl[l].set_fontsize(0.67*fontsize)
            ta.set_ylabel('y (arcsec)', fontsize=fontsize)
            ytl = ta.axes.yaxis.get_majorticklabels()
            for l in range(0, len(ytl)):
                ytl[l].set_fontsize(0.67*fontsize)

        # Show the image data itself.
        if figure_type == 2:
            ta.imshow(tm.data)
            ta.set_axis_off()
            if i == 0:
                fig.text(0.05, 0.2 + i*0.33, differencing_type)


plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.0, 0.0, 1.0, 1.0))
plt.savefig(os.path.expanduser(filepath))
plt.close('all')

fig.text(0, 0.2, 'RD')
fig.text(0, 0.2 + 0.33, 'PD')
fig.text(0, 0.2 + 0.66, 'RDP')
