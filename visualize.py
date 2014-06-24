from __future__ import absolute_import

__authors__ = ["Albert Shih"]
__email__ = "albert.y.shih@nasa.gov"

import numpy as np

def visualize(wave_maps, delay=0.1, range=None, draw_limb=False, draw_grid=False, colorbar=False, vert_line=None):
    """
    Visualizes a list of SunPy Maps.  Requires matplotlib 1.1
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig = plt.figure()
    
    axes = fig.add_subplot(111)
    axes.set_xlabel('X-position [' + wave_maps[0].units['x'] + ']')
    axes.set_ylabel('Y-position [' + wave_maps[0].units['y'] + ']')

    # Warning!  More than a bit hack like.
    # Draw the limb if required
    if draw_limb:
        wave_maps[0].draw_limb(axes=axes)
    # Draw a grid if required
    if draw_grid:
        wave_maps[0].draw_grid(axes=axes)

    extent = wave_maps[0].xrange + wave_maps[0].yrange
    axes.set_title("%s %s" % (wave_maps[0].name, wave_maps[0].date))
    params = {
        "cmap": wave_maps[0].cmap,
        "norm": wave_maps[0].mpl_color_normalizer
    }
    if range != None:
        params["norm"] = colors.Normalize(range[0],range[1])
    img = axes.imshow(wave_maps[0], origin='lower', extent=extent, **params)
    if colorbar:
        fig.colorbar(img)
    if vert_line is not None:
        axes.axvline(vert_line[0])
    fig.show()
    
    for current_wave_map in wave_maps[1:]:
        axes.set_title("%s %s" % (current_wave_map.name, current_wave_map.date))
        if vert_line is not None:
            axes.axvline(vert_line[0])
        img.set_data(current_wave_map)
        plt.pause(delay)

def visualize_dc(wave_maps, delay=0.1, range=None, cmap=None, mpl_color_normalizer=None):
    """
    Visualizes a datacube.  Requires matplotlib 1.1
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig = plt.figure()
    
    axes = fig.add_subplot(111)
    axes.set_xlabel('X-position')
    axes.set_ylabel('Y-position')
    
    params = {
        "cmap": cmap,
        "norm": mpl_color_normalizer
    }
    if range != None:
        params["norm"] = colors.Normalize(range[0],range[1])
    img = axes.imshow(wave_maps[:, :, 0], origin='lower', aspect='auto', **params)
    fig.colorbar(img)
    fig.show()
    nt = wave_maps.shape[2]
    for i in np.arange(1, nt):
        current_wave_map = wave_maps[:, :, i]
        axes.set_title("%i" % (i))
        img.set_data(current_wave_map)
        plt.pause(delay)
