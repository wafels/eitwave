from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE
from sunpy.coordinates.great_arc import great_arc

# Load in a map
m = sunpy.map.Map(AIA_171_IMAGE)

aa = SkyCoord(0*u.deg, -20*u.deg, frame='heliographic_stonyhurst')
bb = SkyCoord(0*u.deg, 20*u.deg, frame='heliographic_stonyhurst')
cc = SkyCoord(60*u.deg, -20*u.deg, frame='heliographic_stonyhurst')
dd = SkyCoord(60*u.deg, 20*u.deg, frame='heliographic_stonyhurst')

a = aa.transform_to(m.coordinate_frame)
b = bb.transform_to(m.coordinate_frame)
c = cc.transform_to(m.coordinate_frame)
d = dd.transform_to(m.coordinate_frame)

aacc = great_arc(aa, cc)
bbdd = great_arc(bb, dd)
aabb = great_arc(aa, bb)
ccdd = great_arc(cc, dd)

ac = great_arc(a, c)
bd = great_arc(b, d)
ab = great_arc(a, b)
cd = great_arc(c, d)

plt.ion()
ax = plt.subplot(projection=m)
m.plot()
m.draw_grid()
ax.plot_coord(ac, color='b')
ax.plot_coord(bd, color='b')
ax.plot_coord(cd, color='b')
ax.plot_coord(ab, color='b')
ax.plot_coord(aacc, color='r')
ax.plot_coord(bbdd, color='r')
ax.plot_coord(ccdd, color='r')
ax.plot_coord(aabb, color='r')

plt.colorbar()
plt.show()


