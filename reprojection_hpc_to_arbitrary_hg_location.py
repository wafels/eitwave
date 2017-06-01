import copy
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE
import numpy as np
from astropy.wcs import WCS

from reproject import reproject_interp

# Read in a map
aia = sunpy.map.Map(AIA_171_IMAGE)

# The number of pixels in the output
shape_out = [360,720]
#wcs_out = aia.wcs.deepcopy()
# Create an empty WCS object with two axes
wcs_out = WCS(naxis=2)

# Define the axis types
wcs_out.wcs.ctype = ['HGLN-CAR', 'HGLT-CAR']

# Define the units of the axes
wcs_out.wcs.cunit = ['deg', 'deg']

# Define the size of each pixel
wcs_out.wcs.cdelt = [0.5, 0.5]

wcs_out.wcs.crval = [80.0, 40.0]

#
wcs_out.wcs.crpix = np.array(shape_out)[::-1]/2.

# Rotation (identity maxtrix means no rotation)
wcs_out.wcs.pc = np.identity(2)


# Do the reprojection
output, footprint = reproject_interp((aia.data, aia.wcs), wcs_out, shape_out)


out_header = copy.deepcopy(aia.meta)
out_header.update(wcs_out.to_header())

# Create the output map
outmap = sunpy.map.Map((output, out_header))

# Plot
fig = plt.figure()
ax = plt.subplot(projection=outmap)
outmap.plot()
lon, lat = ax.coords
lon.set_major_formatter("d.d")
lat.set_major_formatter("d.d")
plt.show()
