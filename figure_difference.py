#
# Make a figure that shows the differences between running difference,
# percentage base difference, running difference persistence images.
#

import os
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from sunpy.map import Map
import mapcube_tools
import aware
import aware_utils
import demonstration_info
from sunpy.time import parse_time
from sunpy import config
TIME_FORMAT = config.get("general", "time_format")
#
def square_root(m):
    data = m.data
    data[data < 0] = -np.sqrt(-data[data < 0])
    data[data > 0] = np.sqrt(data[data > 0])
    # Create a new map object
    return Map(data, m.meta)

plt.ion()

examples = ['previous1', 'corpita_fig7', 'corpita_fig8a']

# Load in all the special information needed
info = demonstration_info.info

# Where the data is
root = os.path.expanduser('~/Data/eitwave')

pbdloc = os.path.expanduser('~/eitwave/')

fig = plt.figure(figsize=(15,10))

thismap = 10

figure_numbering = [['(a)', '(d)', '(g)'], ['(b)', '(e)', '(h)'], ['(c)', '(f)', '(i)']]
wave_numbering = ['Wave A', 'Wave B', 'Wave C']
analysis_type = ['PBD', 'RD', 'RDPI']


for i, example in enumerate(examples):

    fn = figure_numbering[i]

    # Image files
    imgloc = os.path.join(root, 'fts', example)
    
    # Get the file list
    l = aware_utils.get_file_list(imgloc, 'fts')
    
    #
    # Calculate the running difference images
    #
    accum = info[example]["accum"]
    mc = Map(aware_utils.accumulate_from_file_list(l, accum=accum), cube=True)
    
    rd = mapcube_tools.running_difference(mc)
    #
    # Calculate the percentage base difference
    #
    pbd_location = os.path.expanduser(os.path.join(pbdloc, info[example]["pbd"]))
    base = Map(pbd_location).superpixel((4,4))
    base = Map(base.data / base.exposure_time, base.meta)
    pbd = Map(mapcube_tools.base_difference(mc, base=base, fraction=True)[1:], cube=True)
    newpbd = []
    for m in pbd:
        m.data[np.isnan(m.data)] = 0.0
        m.data[np.isinf(m.data)] = 0.0
        newpbd.append(Map(m.data, m.meta))
    pbd = Map(newpbd, cube=True)

    # Get a nicely formatted time
    
    time = '\n{date:{tmf}}'.format(date=parse_time(newrd.date), tmf=TIME_FORMAT)

    #
    # Calculate the running difference persistence images
    #
    rdpi = mapcube_tools.running_difference(mapcube_tools.persistence(mc))

    # Plot the percentage base difference
    a = fig.add_subplot(3, 3, 3 + i + 1)
    cmap = plt.get_cmap("Greys_r")
    pbd[thismap].plot(cmap=cmap)
    if example == 'previous1':
        plt.clim(-0.1, 0.5)
    else:
        plt.clim(0.0, 7.0)
    plt.title(fn[1] + ' ' + wave_numbering[i] + ', ' + analysis_type[1] + time)

    # Plot the running difference
    #Scale the data
    newrd = square_root(rd[thismap])

    a = fig.add_subplot(3, 3, i + 1)
    cmap = plt.get_cmap("Greys_r")
    newrd.plot(cmap=cmap)
    plt.clim(-20.0, 20.0)
    plt.title(fn[0] + ' ' + wave_numbering[i] + ', ' + analysis_type[0] + time)

    # Plot the running difference persistence images
    newrdpi = square_root(rdpi[thismap])
    a = fig.add_subplot(3, 3, 6 + i + 1)
    cmap = plt.get_cmap("Greys_r")
    newrdpi.plot(cmap=cmap)
    plt.title(fn[2] + ' ' + wave_numbering[i] + ', ' + analysis_type[2] + time)

plt.tight_layout()
plt.savefig(os.path.expanduser('~/eitwave-paper/differences.eps'))
