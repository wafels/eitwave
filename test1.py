#
# Trying to remove a lot of extraneous structure in the detection of
# EIT / EUV waves
#
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sunpy.net import hek
from sunpy.map import Map
import aware_utils
from visualize import visualize_dc

# Examples to look at
example = 'previous1'
#example = 'corpita_fig7'

flarelist = {"previous1": {"tr": hek.attrs.Time('2011-10-01 08:56:00', '2011-10-01 10:17:00')},
             "corpita_fig7": {"tr": hek.attrs.Time('2011-02-13 17:32:48', '2011-02-13 17:48:48')}}

# Where the data is
root = os.path.expanduser('~/Data/eitwave')
# Image files
imgloc = os.path.join(root, 'fts', example)
# HEK flare results
pickleloc = os.path.join(root, 'pkl', example)
hekflarename = example + '.hek.pkl'
pkl_file_location = os.path.join(pickleloc, hekflarename)

if not os.path.exists(pickleloc):
    os.makedirs(pickleloc)
    hclient = hek.HEKClient()
    tr = flarelist[example]["tr"]
    ev = hek.attrs.EventType('FL')
    result = hclient.query(tr, ev, hek.attrs.FRM.Name == 'SSW Latest Events')
    pkl_file = open(pkl_file_location, 'wb')
    pickle.dump(result, pkl_file)
    pkl_file.close()
else:
    pkl_file = open(pkl_file_location, 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()

# Get the file list
l = aware_utils.loaddata(imgloc, 'fts')

# Increase signal to noise ratio
print example + ': Accumulating images'
accum = 1
mc = aware_utils.accumulate(l, accum=accum)

# Convert to a datacube
dc = aware_utils.get_datacube(mc)

# Get a persistance datacube
dc2 = aware_utils.persistance_cube(dc)

# Running difference of the persistance datacube
rdc = aware_utils.running_diff_cube(dc2)

# There should be no elements below zero, but just to be sure.
rdc[rdc <= 0] = 0

# Square root to decrease the dynamic range, make the plots look good
rdc2 = np.sqrt(rdc)

# Make a copy for comparison purposes
rdc3 = rdc2.copy()

# Set a noise threshold
noise_threshold = 25.0 * accum
rdc3[rdc3 > noise_threshold] = 0

hhh = jjj

# Animate the datacube
#visualize_dc(rdc3)

# Get the location of the source event
params = aware_utils.params(result[1])

# Convert the datacube into a mapcube
nt = rdc3.shape[2]
rdc3_mapcube = []
for i in range(0, nt):
    new_map = Map(rdc3[:, :, i], mc[i].meta)
    rdc3_mapcube.append(new_map)

# Unravel the mapcube
print example + ': unraveling maps'
urdc3 = aware_utils.map_unravel(rdc3_mapcube, params)

# Animate the mapcube
visualize(urdc3)


"""
get rid of salt and pepper noise now.

From 

http://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html#example-applications-plot-morphology-py

'Morphological opening on an image is defined as an erosion followed by a dilation. Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks.'

from skimage.morphology import opening, disk
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
