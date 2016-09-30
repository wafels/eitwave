
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import astropy.units as u
import sunpy.map
import matplotlib.pyplot as plt

smap = sunpy.map.Map('/Users/ireland/sunpy/data/sample_data/AIA20110319_105400_0171.fits')

smap.peek()

smap2 = smap.superpixel((1, 8)*u.pix)

smap2.peek()

fig, ax = plt.subplots()

im = smap2.plot()

# Prevent the image from being re-scaled while overplotting.
ax.set_autoscale_on(False)

ax.set_title('No WCSAxes')
plt.colorbar()
plt.show()
