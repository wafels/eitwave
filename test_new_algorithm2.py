#
# Trying to remove a lot of extraneous structure in the detection of
# EIT / EUV waves
#
import os
from copy import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from sunpy.net import hek
from sunpy.map import Map
import AWARE

import aware_utils
from visualize import visualize_dc, visualize

plt.ion()

# Examples to look at
example = 'previous1'
#example = 'corpita_fig4'
#example = 'corpita_fig6'
#example = 'corpita_fig7'
#example = 'corpita_fig8a'
#example = 'corpita_fig8e'

info = {"previous1": {"tr": hek.attrs.Time('2011-10-01 08:56:00', '2011-10-01 10:17:00'),
                      "accum": 1,
                      "result": 1,
                      "lon_index": 30},
             "corpita_fig4": {"tr": hek.attrs.Time('2011-06-07 06:15:00', '2011-06-07 07:00:00'),
                               "accum": 2,
                               "result": 0},
             "corpita_fig6": {"tr": hek.attrs.Time('2011-02-08 21:10:00', '2011-02-08 21:21:00'),
                               "accum": 1},
             "corpita_fig7": {"tr": hek.attrs.Time('2011-02-13 17:32:48', '2011-02-13 17:48:48'),
                               "accum": 2,
                               "result": 0},
             "corpita_fig8a": {"tr": hek.attrs.Time('2011-02-15 01:48:00', '2011-02-15 02:14:24'),
                               "accum": 3,
                               "result": 0,
                               "lon_index": 23},
             "corpita_fig8e": {"tr": hek.attrs.Time('2011-02-16 14:22:36', '2011-02-16 14:39:48'),
                               "accum": 3,
                               "result": 0,
                               "lon_index": 5}}

# Where the data is
root = os.path.expanduser('~/Data/eitwave')
# Image files
imgloc = os.path.join(root, 'fts', example)

"""
# HEK flare results
print('Getting HEK flare results.')
pickleloc = os.path.join(root, 'pkl', example)
hekflarename = example + '.hek.pkl'
pkl_file_location = os.path.join(pickleloc, hekflarename)

if not os.path.exists(pickleloc):
    os.makedirs(pickleloc)
    hclient = hek.HEKClient()
    tr = info[example]["tr"]
    ev = hek.attrs.EventType('FL')
    result = hclient.query(tr, ev, hek.attrs.FRM.Name == 'SSW Latest Events')
    pkl_file = open(pkl_file_location, 'wb')
    pickle.dump(result, pkl_file)
    pkl_file.close()
else:
    pkl_file = open(pkl_file_location, 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
"""

# Get the file list
l = aware_utils.get_file_list(imgloc, 'fts')

# Increase signal to noise ratio
print example + ': Accumulating images'
accum = info[example]["accum"]
mc = Map(aware_utils.accumulate_from_file_list(l, accum=accum), cube=True)

#
# new aware
#
transformed = AWARE.processing(mc)


# Get the location of the source event
params = aware_utils.params(result[info[example]['result']])

# Unravel the data
umc = Map(aware_utils.map_unravel(transformed, params), cube=True)

# Get the dynamics of the wave front
# dynamics = AWARE.dynamics(umc, ???)

# Animate the datacube.
# The result of this datacube is the estimated location of the bright front
# as it moves and brightens new pixels as compared to the previous pixels.  If
# the wave does not move but the brightness increases, this will be detected,
# but there will be no apparent motion.  If the wave moves but does not
# increase the brightness in the new region, then the wave will not be
# detected - there is nothing to detect in this case since there is no way
# to know from the brightness that a wave has gone past.

#visualize_dc(rdc3)
#visualize(prdc3, draw_limb=True, draw_grid=True)


"""
# Write some movies
for output in ['original', 'detection']:
    name = example + '.' + output
    FFMpegWriter = animation.writers['ffmpeg']
    fig = plt.figure()
    metadata = dict(title=name, artist='Matplotlib', comment='AWARE test_new_algorithm.py')
    writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=2000.0)
    with writer.saving(fig, 'output_movie.' + name + '.mp4', 100):
        for i in range(1, nt):
            if output == 'original':
                mc[i].plot()
                mc[i].draw_limb()
                mc[i].draw_grid()
                plt.title(mc[i].date)
            if output == 'detection':
                prdc3.maps[i].plot()
                prdc3.maps[i].draw_limb()
                prdc3.maps[i].draw_grid()
                plt.title(mc[i].date)
            writer.grab_frame()
"""



# Get the location of the source event
params = aware_utils.params(result[info[example]['result']])

# Convert the datacube into a mapcube
#nt = rdc3.shape[2]
#rdc3_mapcube = []
#for i in range(0, nt):
#    new_map = Map(rdc3[:, :, i], mc[i].meta)
#    rdc3_mapcube.append(new_map)
# Unravel the mapcube - the unravel appears to work when using sunpy master on
# my home machine.

print example + ': unraveling cleaned maps.'
uprdc3 = aware_utils.map_unravel(closing_cleaned, params)
# Final unraveled datacube, with time in the first dimension
#dfinal = np.asarray([m.data for m in uprdc3])
dfinal = np.asarray([m.data for m in closing_cleaned])

#print example + ': unraveling data maps.'
#umc = aware_utils.map_unravel(mc, params)
#dumc = np.asarray([m.data for m in umc])
# Animate the mapcube
# visualize(uprdc3)
# Show the evolution of the wavefront at a single longitude.
lon_index = info[example]["lon_index"]

# Plot out a map
visualize([uprdc3[10]], vert_line=[-180 + lon_index* params.get('lon_bin')])




timescale = accum * 12

plt.figure(1)
plt.imshow(dfinal[:, :, lon_index], aspect='auto', extent=[0, dfinal.shape[1] * params.get('lat_bin'), 0, dfinal.shape[0] * timescale], origin='bottom')
plt.ylabel('time (seconds) after ' + mc[0].date)
plt.xlabel('latitude')
plt.title('Wave front at longitude = %f' % (lon_index * params.get('lon_bin')))
plt.show()

#lt.figure(2)
#plt.imshow(np.sqrt(dumc[:, :, lon_index]), aspect='auto', extent=[0, dumc.shape[1] * params.get('lat_bin'), 0, dumc.shape[0] * timescale], origin='bottom')
#plt.ylabel('elapsed time (seconds) after ' + mc[0].date)
#plt.xlabel('latitude')
#plt.title('Wave front at longitude = %f' % (lon_index * params.get('lon_bin')))
#plt.show()

# At all times get an average location of the wavefront
latitude = np.min(closing_cleaned[10].yrange) + np.arange(0, dfinal.shape[1]) * params.get('lat_bin')
loc = np.zeros(nt)
std = np.zeros_like(loc)
for i in range(0, nt):
    loc[i] = np.sum(dfinal[i, :, lon_index] * latitude) / np.sum(dfinal[i, :, lon_index])
    std[i] = np.std(dfinal[i, :, lon_index] * latitude / np.sum(dfinal[i, :, lon_index]))

# Do a quadratic fit to the data
isfinite = np.isfinite(loc)
time = timescale * np.arange(0, nt)
timef = time[isfinite][1:]
locf = loc[isfinite][1:]
locf = np.abs(locf - locf[0])
stdf = std[isfinite][1:]
quadfit = np.polyfit(timef, locf, 2, w=stdf)
bestfit = np.polyval(quadfit, timef)

factor = 1.21e4 # circumference of the Sun divided by its radius.
vel = round(quadfit[1] * factor, 1)
acc = round(quadfit[0] * factor, 1)

plt.figure(3)
plt.errorbar(timef, locf, yerr=stdf, fmt='go', label='measured wavefront position')
plt.plot(timef, bestfit, label='quadratic fit')
plt.title('wavefront position')
plt.xlabel('elapsed time (seconds) after ' + mc[0].date)
plt.ylabel('degrees traversed relative to launch site')
xpos = 0.65 * np.max(timef)
ypos = [np.min(locf) + 0.6 * (np.max(locf) - np.min(locf)), np.min(locf) + 0.7 * (np.max(locf) - np.min(locf))]
plt.annotate(r'v = '+ str(vel) + ' $km/s$', [xpos, ypos[0]], fontsize=20)
plt.annotate(r'a = '+ str(acc) + ' $km/s^{2}$', [xpos, ypos[1]], fontsize=20)
plt.ylim(0, 1.3 * np.max(locf))
plt.legend()






# Convert this in to a datacube and get the cross sections out.  Super simple
# to fit with Gaussians.  Note what we are measuring - it is the location of
# what lit up next in the corona.  The actual bright front could be a lot
# wider.

"""
get rid of salt and pepper noise now.

From 

http://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html#example-applications-plot-morphology-py

from skimage.filter.rank import median

i = 7
z = rdc3[:, :, i] / rdc3[:, :, i].max()
plt.imshow(opening(z, disk(3)))
plt.imshow(median(z, disk(11)))

plt.imshow(closing(median(rdc3[:,:,i]/rdc3.max(), disk(11)),disk(11)))

median filtering : from skimage.filter.rank import median
bilateral filtering

http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_filtering/py_filtering.html

"""


#
# After this, perhaps we can do some morphological operations to join
# neighbouring weak pixels.  Perhaps also do a 3-d morphology to take
# advantage of previous and future observations.
#

"""
# see the wave
plt.imshow(np.sqrt(dc2[:,:,7]) - np.sqrt(dc2[:,:,0]))

# and again
plt.imshow(np.sqrt(np.abs(dc2[:,:,7]-dc2[:,:,0])))

# one more time.
med = np.median(dc, axis=2)
plt.imshow(np.sqrt(np.abs(dc2[:,:,7]-med)))

# this is cool
plt.imshow(np.sqrt(np.abs(dc2[:,:,7]-dc2[:,:,6])))
plt.imshow(np.sqrt(np.abs(dc2[:,:,6]-dc2[:,:,5])))
plt.imshow(np.sqrt(np.abs(dc2[:,:,5]-dc2[:,:,4])))
plt.imshow(np.sqrt(np.abs(dc2[:,:,4]-dc2[:,:,3])))
plt.imshow(np.sqrt(np.abs(dc2[:,:,5]-dc2[:,:,4])))
"""


"""
Used these full disk images acquired via the cutout service.

['/Users/ireland/aware_data/fts/ssw_cutout_20111001_095002_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095026_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095050_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095114_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095127_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095138_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095150_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095202_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095226_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095250_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095314_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095338_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095402_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095426_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095439_AIA_211_.fts',
 '/Users/ireland/aware_data/fts/ssw_cutout_20111001_095450_AIA_211_.fts']
"""

"""
example1 : http://helioviewer.org/?date=2011-10-01T09:58:48.000Z&imageScale=2.4204409&centerX=0&centerY=0&imageLayers=%5BSDO,AIA,AIA,171,1,100%5D&eventLayers=%5BFL,SSW_Latest_Events,1%5D&eventLabels=true
corpita_fig7 : http://helioviewer.org/?date=2011-02-13T17:30:00.000Z&imageScale=2.4204409&centerX=1.2102943159942627&centerY=0.4840438604034424&imageLayers=%5BSDO,AIA,AIA,171,1,100%5D&eventLayers=%5BFL,SSW_Latest_Events,1%5D&eventLabels=true
"""
"""
from sunpy.net import vso
client=vso.VSOClient()
qr = client.query(vso.attrs.Time('2011/02/08 21:10:00', '2011/02/08 21:21:00'), vso.attrs.Instrument('aia'), vso.attrs.Wave(211,211))
res = client.get(qr, path="{file}.fts")
"""


"""
# median filter, then morphological operation (closing).
print('Applying filters.')
closing_radius = 11
prdc3 = []
for i in range(0, nt):
    # Normalize
    img = rdc3[:, :, i] / rdc3[:, :, i].max()
    # Clean up the noise
    img = median(img, disk(median_radius))
    # Join up the detection
    img = closing(img, disk(closing_radius))
    # Create a sunpy map and put it into a list
    prdc3.append(Map(img, dc_meta[i + 1]))

# Convert to a mapcube
prdc3 = Map(prdc3, cube=True)
"""