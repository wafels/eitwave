import numpy as np
import aware_utils
import matplotlib.pyplot as plt
from visualize import visualize_dc

l = aware_utils.loaddata('/home/ireland/Data/eitwave/fts/dir1/', 'fts')
l = aware_utils.loaddata('/home/ireland/Data/eitwave/fts/corpita_fig7/', 'fts')

accum = 2
mc = aware_utils.accumulate(l, accum=accum)
dc = aware_utils.get_datacube(mc)
dc2 = aware_utils.persistance_cube(dc)
rdc = aware_utils.running_diff_cube(dc2)
rdc[rdc <= 0] = 0
rdc2 = np.sqrt(rdc)
rdc3 = rdc2.copy()
rdc3[rdc3 >(25.0 * accum)]=0
#
# Trying to remove a lot of extraneous structure
#

visualize_dc(rdc3)

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
